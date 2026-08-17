"""
train_lora.py — bf16 LoRA SFT of a causal LLM as a target-conditioned Arabic stance classifier.

NEW FAMILY (idea 2): a generative LLM (ALLaM-7B / Qwen2.5-7B) fine-tuned as a stance
classifier via *candidate-scoring*. Design (validated by 2 frontier-model consults):
  * Arabic MCQ prompt, single-letter answer  أ=Favor  ب=Against  ج=None
    (all three letters are SINGLE tokens in both tokenizers -> clean 3-way softmax).
  * Training target = that one answer letter (+eos); prompt tokens masked (loss on answer only).
  * Inference proba = softmax over the 3 answer-letter logits at the answer position,
    reordered to stance_lib column order [Against, Favor, None] so it blends directly
    with the encoder OOF proba (results/*/oof_proba.npz).
  * bf16 LoRA (NOT 4-bit) — H100 has the memory; simpler + slightly higher quality than QLoRA,
    and avoids the bitsandbytes/CC-wheelhouse mess.

Outputs under --out_dir (== $EXP_OUTPUT_DIR under SLURM):
  oof_proba.npz  (proba[N,3] in eval-CSV row order, y_true, target)   <- the CV signal
  metrics.json   (best-epoch Favg2/Favg3/Acc + per-target Favg2)
  adapter/       (best-epoch LoRA adapter, for deployment)
"""
import os, sys, json, argparse, random, time
os.environ.setdefault("HF_HOME", "<cache>/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

# --------------------------------------------------------------------------- #
# Verbalizer: class-id -> answer letter.  Prompt lists options in fixed A/B/C
# order (Favor/Against/None); we gather each id's letter logit into column=id.
# stance_lib ids: Against=0, Favor=1, None=2.
# --------------------------------------------------------------------------- #
LETTERS = {
    "ar_letter": {1: "أ", 0: "ب", 2: "ج"},   # Favor=أ  Against=ب  None=ج
    "lat_letter": {1: "A", 0: "B", 2: "C"},
    "digit": {1: "1", 0: "2", 2: "3"},        # 3rd prompt (different output space, for ensemble diversity)
}
OPTION_LINES = {
    "ar_letter": "أ) مؤيد للهدف\nب) معارض للهدف\nج) محايد أو لا يذكر موقفاً",
    "lat_letter": "A) Favor (مؤيد)\nB) Against (معارض)\nC) None (محايد)",
    "digit": "1) مؤيد للهدف\n2) معارض للهدف\n3) محايد أو لا يذكر موقفاً",
}
# Arabic gloss for the SEEN targets (fallback = raw string for any unseen target).
TARGET_AR = {
    "Women empowerment": "تمكين المرأة",
    "Covid Vaccine": "لقاح فيروس كورونا (كوفيد-19)",
    "Digital Transformation": "التحول الرقمي",
    # Track-1 blind TEST target (held-out, unseen in training) — Saudi women-driving topic.
    "Women Driving": "قيادة المرأة",
}


def prompt_to_ids(tok, prompt, max_len):
    """Robustly return List[int] of the chat-formatted prompt (up to the assistant turn)."""
    out = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True, return_dict=False)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if len(out) and isinstance(out[0], (list, tuple)):
        out = out[0]
    out = list(out)
    if len(out) > max_len - 2:
        out = out[: max_len - 2]
    return out


def build_prompt(text, target, verb):
    tgt = TARGET_AR.get(target, target)
    return (
        "صنّف موقف كاتب التغريدة التالية تجاه الهدف المُحدَّد.\n\n"
        f"التغريدة: {text}\n"
        f"الهدف: {tgt}\n\n"
        "اختر الإجابة الصحيحة:\n"
        f"{OPTION_LINES[verb]}\n\n"
        "أجب بحرف واحد فقط."
    )


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class StanceDS(Dataset):
    """Tokenized (prompt-masked) training examples with optional target-swap NONE negatives."""
    def __init__(self, df, tok, verb, max_len, swap_neg=0.0, seed=0, is_train=True,
                 conf_weight="none", conf_mean=1.0):
        self.rows = []            # (input_ids, labels, w) for train ; (prompt_ids, y, target) for eval
        self.tok = tok
        self.max_len = max_len
        self.is_train = is_train
        letter_id = LETTERS[verb]
        eos = tok.eos_token_id
        all_targets = sorted(df[S.TARGET_COL].unique().tolist())
        rng = random.Random(seed)
        n_trunc = 0
        for _, r in df.iterrows():
            text, target, y = r["text_proc"], r[S.TARGET_COL], int(r["label"])
            # per-instance loss weight (normalized so the batch loss scale matches uniform)
            w = (float(r["conf"]) / conf_mean) if conf_weight == "linear" else 1.0
            pids = prompt_to_ids(tok, build_prompt(text, target, verb), max_len)
            if is_train:
                ans = tok.encode(letter_id[y], add_special_tokens=False) + [eos]
                self.rows.append((pids + ans, [-100] * len(pids) + ans, w))
                # target-swap hard negative -> NONE toward an unrelated target
                if swap_neg > 0 and rng.random() < swap_neg:
                    others = [t for t in all_targets if t != target]
                    if others:
                        p2ids = prompt_to_ids(tok, build_prompt(text, rng.choice(others), verb), max_len)
                        ans2 = tok.encode(letter_id[2], add_special_tokens=False) + [eos]  # NONE
                        self.rows.append((p2ids + ans2, [-100] * len(p2ids) + ans2, 1.0))
            else:
                self.rows.append((pids, y, target))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate_train(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    ids, lab, att, wts = [], [], [], []
    for input_ids, labels, w in batch:
        n = maxlen - len(input_ids)
        ids.append(input_ids + [pad_id] * n)
        lab.append(labels + [-100] * n)
        att.append([1] * len(input_ids) + [0] * n)
        wts.append(w)
    return (torch.tensor(ids), torch.tensor(att), torch.tensor(lab),
            torch.tensor(wts, dtype=torch.float32))


@torch.no_grad()
def candidate_score(model, tok, eval_ds, verb, device, batch=16):
    """proba[N,3] via softmax over the 3 answer-letter logits at the answer position."""
    model.eval()
    letter_id = LETTERS[verb]
    # token id for each class's letter (single token — verified in probe)
    col_tok = {}
    for cid, ltr in letter_id.items():
        t = tok.encode(ltr, add_special_tokens=False)
        assert len(t) == 1, f"verbalizer letter {ltr!r} is not a single token: {t}"
        col_tok[cid] = t[0]
    order = [col_tok[0], col_tok[1], col_tok[2]]  # gather order -> columns [Against,Favor,None]
    pad_id = tok.pad_token_id
    N = len(eval_ds)
    proba = np.zeros((N, 3), dtype=np.float32)
    ys = np.zeros(N, dtype=np.int64)
    tgts = []
    for i in range(0, N, batch):
        chunk = [eval_ds[j] for j in range(i, min(i + batch, N))]
        prompts = [c[0] for c in chunk]
        maxlen = max(len(p) for p in prompts)
        # LEFT-pad for generation so the last position is the true answer slot for every row
        ids, att, lastpos = [], [], []
        for p in prompts:
            n = maxlen - len(p)
            ids.append([pad_id] * n + p)
            att.append([0] * n + [1] * len(p))
            lastpos.append(maxlen - 1)
        ids = torch.tensor(ids, device=device); att = torch.tensor(att, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=att).logits
        last = logits[torch.arange(len(chunk)), torch.tensor(lastpos, device=device)].float()
        sub = last[:, order]                       # [b,3] logits for [Against,Favor,None]
        pr = torch.softmax(sub, dim=1).cpu().numpy()
        proba[i:i + len(chunk)] = pr
        for k, c in enumerate(chunk):
            ys[i + k] = c[1]; tgts.append(c[2])
    return proba, ys, np.array(tgts, dtype=object)


def per_target_favg2(proba, y, tgt):
    out = {}
    for t in sorted(set(tgt.tolist())):
        m = tgt == t
        out[t] = round(S.compute_metrics(y[m], proba[m].argmax(1))["Favg2"] * 100, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_csv", default=S.TRAIN_CSV)
    ap.add_argument("--aug_csv", default="", help="extra labeled rows (minority-targeted aug) appended to train")
    ap.add_argument("--eval_csv", default=S.DEV_CSV)
    ap.add_argument("--out_dir", default=os.environ.get("EXP_OUTPUT_DIR", "results/lora_dev"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_modules", default="attn", choices=["attn", "all"])
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--verbalizer", default="ar_letter", choices=list(LETTERS))
    ap.add_argument("--preprocess", default="light")
    ap.add_argument("--swap_neg", type=float, default=0.0)
    ap.add_argument("--conf_weight", default="none", choices=["none", "linear"],
                    help="per-instance loss weighting by stance:confidence (linear: w_i=conf/mean; "
                         "down-weights ambiguous/label-noise boundary rows). Distinct from uniform LS (E3).")
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="subset N train rows (sanity)")
    ap.add_argument("--eval_limit", type=int, default=0)
    ap.add_argument("--save_adapter", type=int, default=1)
    ap.add_argument("--save_last", type=int, default=0, help="deploy: save the FINAL-epoch adapter (fixed-epoch, no eval-based selection)")
    args = ap.parse_args()
    print("[args]", json.dumps(vars(args), ensure_ascii=False), flush=True)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    tr = S.load_labeled(args.train_csv, preprocess=args.preprocess)
    if args.aug_csv and os.path.exists(args.aug_csv):
        aug = S.load_labeled(args.aug_csv, preprocess=args.preprocess)
        pre = len(tr)
        tr = pd.concat([tr, aug], ignore_index=True)
        print(f"[aug] +{len(aug)} rows from {os.path.basename(args.aug_csv)} -> train {pre}->{len(tr)} "
              f"| stance dist:\n{tr.groupby([S.TARGET_COL,'label']).size().unstack(fill_value=0)}", flush=True)
    ev = S.load_labeled(args.eval_csv, preprocess=args.preprocess)
    if args.limit:
        tr = tr.sample(n=min(args.limit, len(tr)), random_state=args.seed).reset_index(drop=True)
    if args.eval_limit:
        ev = ev.iloc[: args.eval_limit].reset_index(drop=True)
    print(f"[data] train={len(tr)} eval={len(ev)} targets={sorted(tr[S.TARGET_COL].unique())}", flush=True)

    conf_mean = float(tr["conf"].mean()) if args.conf_weight == "linear" else 1.0
    if args.conf_weight == "linear":
        print(f"[conf] loss-weighting by stance:confidence (mean={conf_mean:.4f} min={tr['conf'].min():.3f}) "
              f"— down-weights ambiguous boundary rows", flush=True)
    train_ds = StanceDS(tr, tok, args.verbalizer, args.max_len, swap_neg=args.swap_neg,
                        seed=args.seed, is_train=True, conf_weight=args.conf_weight, conf_mean=conf_mean)
    eval_ds = StanceDS(ev, tok, args.verbalizer, args.max_len, is_train=False)
    print(f"[data] train_examples(incl swap)={len(train_ds)} eval={len(eval_ds)}", flush=True)

    modules = (["q_proj", "k_proj", "v_proj", "o_proj"] if args.lora_modules == "attn"
               else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=modules, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for _, p in model.named_parameters():          # LoRA master weights in fp32 for stable AdamW
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.base_model} | trainable params={ntr/1e6:.2f}M | modules={modules}", flush=True)

    dl = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True,
                    collate_fn=lambda b: collate_train(b, tok.pad_token_id), drop_last=False)
    steps_per_epoch = (len(dl) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    print(f"[opt] microbatches/epoch={len(dl)} optim_steps/epoch={steps_per_epoch} total={total_steps}", flush=True)

    best = None
    best_favg2 = -1.0
    best_proba = None
    all_proba = {}          # per-epoch OOF proba -> robust seed-avg + epoch choice offline
    curve = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); run = 0.0; nb = 0
        opt.zero_grad()
        for bi, (ids, att, lab, wts) in enumerate(dl):
            ids, att, lab, wts = ids.to(device), att.to(device), lab.to(device), wts.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if args.conf_weight == "none":
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                else:
                    logits = model(input_ids=ids, attention_mask=att).logits
            if args.conf_weight != "none":
                # per-example weighted CE over the answer tokens (fp32, outside autocast)
                sl = logits[:, :-1, :].float()
                slb = lab[:, 1:]
                tokl = F.cross_entropy(sl.reshape(-1, sl.size(-1)), slb.reshape(-1),
                                       ignore_index=-100, reduction="none").view(slb.size())
                mmask = (slb != -100).float()
                per_ex = (tokl * mmask).sum(1) / mmask.sum(1).clamp(min=1.0)
                loss = (per_ex * wts).sum() / wts.sum().clamp(min=1e-8)
            (loss / args.grad_accum).backward()
            run += loss.item(); nb += 1
            if (bi + 1) % args.grad_accum == 0 or (bi + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                opt.step(); sched.step(); opt.zero_grad()
        proba, ys, tgts = candidate_score(model, tok, eval_ds, args.verbalizer, device, batch=args.eval_batch)
        m = S.compute_metrics(ys, proba.argmax(1))
        ptg = per_target_favg2(proba, ys, tgts)
        print(f"[ep{ep}] loss={run/max(nb,1):.4f} {time.time()-t0:.0f}s | "
              f"Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f} Acc={m['Acc']*100:.2f} "
              f"F_fav={m['F_favor']*100:.1f} F_agn={m['F_against']*100:.1f} | per-target={ptg}", flush=True)
        all_proba[f"proba_ep{ep}"] = proba.copy()
        curve.append({"epoch": ep, "Favg2": round(m["Favg2"] * 100, 3), "per_target": ptg})
        if m["Favg2"] > best_favg2:                     # compare FRACTIONS (bugfix: was vs stored percent)
            best_favg2 = m["Favg2"]
            best = {"epoch": ep, **{k: round(v * 100, 3) for k, v in m.items() if k != "pred"},
                    "per_target": ptg}
            best_proba = proba.copy()
            if args.save_adapter:
                model.save_pretrained(os.path.join(args.out_dir, "adapter"))
                tok.save_pretrained(os.path.join(args.out_dir, "adapter"))

    if args.save_last and args.save_adapter:          # deploy: overwrite best-epoch save with the FINAL epoch
        model.save_pretrained(os.path.join(args.out_dir, "adapter"))
        tok.save_pretrained(os.path.join(args.out_dir, "adapter"))
        print(f"[save_last] deploy: saved FINAL-epoch (ep{args.epochs}) adapter", flush=True)
    np.savez(os.path.join(args.out_dir, "oof_proba.npz"),
             proba=best_proba, y_true=ys, target=tgts, **all_proba)
    json.dump({"base_model": args.base_model, "seed": args.seed, "verbalizer": args.verbalizer,
               "best": best, "curve": curve, "args": vars(args)},
              open(os.path.join(args.out_dir, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[DONE] best {json.dumps(best, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
