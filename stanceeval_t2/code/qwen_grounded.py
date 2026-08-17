"""
qwen_grounded.py — step A: LOCAL zero-shot grounded stance voter.

T1's MCQ candidate-scoring (single Arabic answer-letter softmax أ/ب/ج → clean 3-way
proba) applied ZERO-SHOT (no adapter) to a cached Qwen2.5 Instruct model, with THIS
project's per-target grounding CARD (definition + aspects) injected into the prompt.
Zero API / zero bill / offline (SLURM compute node; models load by name from the SHARED
/project HF cache — HF_HOME is preset by the environment; do NOT override it).

Output npz mirrors the llm voters ({proba SOFT, pred_id, target, idx, y_true}) in
--order file|loto row order → drop-in for blend_eval.py against ENC/claude/gpt5.
"""
import os, sys, json, argparse, time
os.environ.setdefault("HF_HOME", "<cache>/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from train_lora import LETTERS, OPTION_LINES, prompt_to_ids, TARGET_AR

# ZERO-SHOT option lines (canary 07-03): the LoRA option "محايد أو لا يذكر موقفاً" invites
# None on implicit stance when un-finetuned (canary over-predicted None 17 vs gold 7);
# Mawqif None = unrelated/no evaluation, and it's RARE. Zero-shot only — train_lora's
# OPTION_LINES stay as the T1-proven training format.
ZS_OPTION_LINES = dict(OPTION_LINES)
ZS_OPTION_LINES["ar_letter"] = ("أ) مؤيد للهدف (ولو بشكل غير مباشر)\n"
                                "ب) معارض للهدف (ولو بشكل غير مباشر)\n"
                                "ج) لا علاقة للتغريدة بالهدف، أو لا تحمل أي موقف أو تقييم تجاهه")


def build_grounded_prompt(text, target, card, verb):
    tgt = TARGET_AR.get(target, target)
    lines = ["صنّف موقف كاتب التغريدة التالية تجاه الهدف المُحدَّد.", ""]
    lines.append(f"الهدف: {tgt}")
    if card.get("definition"):
        lines.append(f"تعريف الهدف: {card['definition']}")
    if card.get("aspects"):
        lines.append(f"جوانب متصلة بالهدف: {card['aspects']}")
    lines += [
        "",
        f"التغريدة: {text}",
        "",
        "ملاحظات: قد يُعبَّر عن الموقف بشكل غير مباشر عبر جانب من جوانب الهدف — وهذا يُعدُّ موقفاً تجاه الهدف. "
        "السخرية والأسئلة الاستنكارية غالباً تدل على المعارضة. "
        "اختر (ج) فقط إذا كانت التغريدة غير متعلقة بالهدف أو وصفية بحتة دون أي تقييم.",
        "",
        "اختر الإجابة الصحيحة:",
        ZS_OPTION_LINES[verb],
        "",
        "أجب بحرف واحد فقط.",
    ]
    return "\n".join(lines)


@torch.no_grad()
def score(model, tok, prompts, order, pad_id, device, batch=16):
    N = len(prompts)
    proba = np.zeros((N, 3), np.float32)
    t0 = time.time()
    for i in range(0, N, batch):
        chunk = prompts[i:i + batch]
        ml = max(len(p) for p in chunk)
        ids = [[pad_id] * (ml - len(p)) + p for p in chunk]          # LEFT-pad
        att = [[0] * (ml - len(p)) + [1] * len(p) for p in chunk]
        ids = torch.tensor(ids, device=device); att = torch.tensor(att, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(input_ids=ids, attention_mask=att).logits
        last = lg[torch.arange(len(chunk)), ml - 1].float()[:, order]
        proba[i:i + len(chunk)] = torch.softmax(last, 1).cpu().numpy()
        if (i // batch) % 20 == 0:
            print(f"  {i+len(chunk)}/{N} ({time.time()-t0:.0f}s)", flush=True)
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--csv", default=S.DEV_CSV)
    ap.add_argument("--order", default="file", choices=["loto", "file"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cards_from", nargs="+", required=True, help="meta.json files with {'cards': {target: card}}")
    ap.add_argument("--verbalizer", default="ar_letter", choices=list(LETTERS))
    ap.add_argument("--preprocess", default="light")
    ap.add_argument("--max_len", type=int, default=560)  # measured 07-03: dev max 472 / train max 523 grounded tokens — 384 TAIL-TRUNCATED 58-82% of rows (canary bug)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cards = {}
    for f in args.cards_from:
        cards.update(json.load(open(f))["cards"])
    print(f"[cards] {list(cards)}", flush=True)

    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", S.TARGET_COL):
        df[c] = df[c].astype(str).str.strip()
    df["text_proc"] = df["text"].apply(S.PREPROCESSORS[args.preprocess])
    labeled = "stance" in df.columns and (df["stance"].astype(str).str.strip() != "").all()
    if args.order == "loto":
        order_idx = []
        for tg in dict.fromkeys(df[S.TARGET_COL].tolist()):
            order_idx.extend(df.index[df[S.TARGET_COL] == tg].tolist())
    else:
        order_idx = df.index.tolist()
    if args.limit:
        order_idx = order_idx[: args.limit]
    missing = sorted({df.at[i, S.TARGET_COL] for i in order_idx} - set(cards))
    assert not missing, f"no card for targets: {missing}"

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    letter_id = LETTERS[args.verbalizer]
    col_tok = {}
    for cid, ltr in letter_id.items():
        t = tok.encode(ltr, add_special_tokens=False)
        assert len(t) == 1, f"letter {ltr!r} not single-token: {t}"
        col_tok[cid] = t[0]
    order = [col_tok[0], col_tok[1], col_tok[2]]                     # -> [Against,Favor,None]

    prompts = [prompt_to_ids(tok, build_grounded_prompt(
        df.at[i, "text_proc"], df.at[i, S.TARGET_COL], cards[df.at[i, S.TARGET_COL]], args.verbalizer),
        args.max_len) for i in order_idx]
    n_cap = sum(1 for p in prompts if len(p) >= args.max_len - 2)
    print(f"[data] n={len(prompts)} max_prompt_tok={max(len(p) for p in prompts)} "
          f"AT-CAP(tail-truncated)={n_cap}", flush=True)
    assert n_cap <= max(2, len(prompts) // 200), \
        f"{n_cap} prompts hit max_len={args.max_len} — raise --max_len (tail truncation destroys the MCQ options)"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    print(f"[model] {args.base_model} loaded {time.time()-t0:.0f}s", flush=True)
    proba = score(model, tok, prompts, order, tok.pad_token_id, device, args.batch)

    pred_ids = proba.argmax(1)
    tgt = np.array([df.at[i, S.TARGET_COL] for i in order_idx], dtype=object)
    out = {"proba": proba, "pred_id": pred_ids, "target": tgt.astype(str), "idx": np.array(order_idx)}
    res = {"model": args.base_model, "grounded": "zs-local-v2", "n": len(order_idx),
           "pred_dist": np.bincount(pred_ids, minlength=3).tolist(),
           "runtime_sec": round(time.time() - t0, 1)}
    if labeled:
        y = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in order_idx])
        out["y_true"] = y
        res["pooled_Favg2"] = round(S.compute_metrics(y, pred_ids)["Favg2"] * 100, 2)
        res["per_target"] = {tg: round(S.compute_metrics(y[tgt == tg], pred_ids[tgt == tg])["Favg2"] * 100, 2)
                             for tg in dict.fromkeys(tgt.tolist())}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    json.dump(res, open(os.path.splitext(args.out)[0] + "_metrics.json", "w"), ensure_ascii=False, indent=2)
    print("[RESULT] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
