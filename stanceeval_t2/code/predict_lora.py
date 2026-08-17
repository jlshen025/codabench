"""
predict_lora.py — DEPLOY inference for a LoRA generative stance model (e.g. Qwen-14B).
Loads base + one-or-more LoRA adapters (seed-avg), runs the SAME single-letter
candidate-scoring as train_lora, and writes a proba npz aligned to the test-CSV row
order (stance_lib columns [Against,Favor,None]) — directly blendable with the encoder
proba in predict_test.py.

Eval runbook (Track1 held-best = enc(0.4) + Qwen14-LoRA(0.6)):
  1. python predict_lora.py --base_model Qwen/Qwen2.5-14B-Instruct \
       --adapters <q14_traindev_s42/adapter> <q14_traindev_s1/adapter> \
       --test_csv TEST --out_npz scripts/_llm/q14_test.npz
  2. python predict_test.py --models <marbertv2_mtl_full> <araberttw_mtl_full> \
       --test_csv TEST --llm_npz scripts/_llm/q14_test.npz --w_llm 0.6 --out_zip submission/track1.zip
If the CSV has a 'stance' column (e.g. dev.csv), prints Favg2 for validation.
"""
import os, sys, argparse
os.environ.setdefault("HF_HOME", "<cache>/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from train_lora import build_prompt, prompt_to_ids, LETTERS


@torch.no_grad()
def score(model, tok, prompts, order, pad_id, device, batch=16):
    N = len(prompts)
    proba = np.zeros((N, 3), np.float32)
    for i in range(0, N, batch):
        chunk = prompts[i:i + batch]
        ml = max(len(p) for p in chunk)
        ids = [[pad_id] * (ml - len(p)) + p for p in chunk]        # LEFT-pad for gen
        att = [[0] * (ml - len(p)) + [1] * len(p) for p in chunk]
        ids = torch.tensor(ids, device=device); att = torch.tensor(att, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(input_ids=ids, attention_mask=att).logits
        last = lg[torch.arange(len(chunk)), ml - 1].float()[:, order]
        proba[i:i + len(chunk)] = torch.softmax(last, 1).cpu().numpy()
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="LoRA adapter dir(s), seed-averaged")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--verbalizer", default="ar_letter")
    ap.add_argument("--preprocess", default="light")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    df = pd.read_csv(args.test_csv, keep_default_na=False, dtype=str)
    for c in ("text", S.TARGET_COL):
        df[c] = df[c].astype(str).str.strip()
    df["text_proc"] = df["text"].apply(S.PREPROCESSORS[args.preprocess])
    tok = AutoTokenizer.from_pretrained(args.adapters[0])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    letter_id = LETTERS[args.verbalizer]
    col_tok = {}
    for cid, ltr in letter_id.items():
        t = tok.encode(ltr, add_special_tokens=False)
        assert len(t) == 1, f"letter {ltr!r} not single-token: {t}"
        col_tok[cid] = t[0]
    order = [col_tok[0], col_tok[1], col_tok[2]]                    # -> [Against,Favor,None]
    prompts = [prompt_to_ids(tok, build_prompt(r["text_proc"], r[S.TARGET_COL], args.verbalizer), args.max_len)
               for _, r in df.iterrows()]

    P = []
    for ad in args.adapters:
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                    attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, ad).to(device).eval()
        P.append(score(model, tok, prompts, order, tok.pad_token_id, device, args.batch))
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[predict_lora] scored adapter {ad}", flush=True)
    proba = np.mean(P, 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez(args.out_npz, proba=proba.astype(np.float32))
    dist = np.bincount(proba.argmax(1), minlength=3).tolist()
    print(f"[out] {len(df)} rows, {len(args.adapters)} adapter(s) -> {args.out_npz} dist={dist}", flush=True)
    if "stance" in df.columns and (df["stance"].astype(str).str.strip() != "").all():
        y = np.array([S.LABEL2ID[s.strip()] for s in df["stance"]])
        m = S.compute_metrics(y, proba.argmax(1))
        print(f"[VALIDATION vs gold] Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
