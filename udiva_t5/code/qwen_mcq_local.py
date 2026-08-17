#!/usr/bin/env python3
"""UDIVA-HHOI Track 5 — fully local LLM inference for the causal MCQs.

Runs an open-weight instruction-tuned LLM (default: Qwen/Qwen3-32B) locally via
HuggingFace transformers over the causal single-choice questions: one prompt per effect
containing the effect description and the five candidate-cause options, deterministic
(greedy) decoding, answer = one letter A-E. No data leaves the machine.

Usage:
  # development set — scores accuracy against the starting-kit reference:
  python qwen_mcq_local.py --split dev [--limit 50] [--out dev_labels.json]
  # test set — writes the labels consumed by make_submission.py:
  python qwen_mcq_local.py --split test --out test_labels.json

Model requirements: Qwen3-32B needs transformers>=4.51 and ~66 GB VRAM in bf16
(one 80 GB GPU, device_map="auto"); use --model Qwen/Qwen3-8B (~17 GB) for a smaller
GPU. Any local instruction-model id works; a tiny one (e.g. Qwen/Qwen2.5-0.5B-Instruct)
with --limit serves as a fast CPU smoke test of the pipeline.
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import udiva

EVAL_MCQS = "<datasets>/UDIVA-HHOI/evaluation/eval_data/causal_MCQS.json"

SYS = ("You are an expert at causal reasoning about dyadic human-human-object "
       "interactions: two participants (A and B) collaboratively build a Lego model, "
       "with a supervisor. For each EFFECT event you are given 5 candidate CAUSE "
       "descriptions (A-E). Exactly one is the true causal antecedent that happened "
       "BEFORE and brought about the effect; the others are plausible distractors. "
       "Use commonsense about collaboration, joint attention, turn-taking, and the "
       "task to pick the single best cause.")

USER_TMPL = (
    "EFFECT: {desc}\n"
    "{options}\n"
    "Which option (A-E) is the true causal antecedent of the EFFECT? "
    "Reply with ONLY the single letter."
)


def build_messages(e):
    opts = "\n".join(f"{L}) {e.options.get(L, '')}" for L in "ABCDE")
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER_TMPL.format(desc=e.description, options=opts)},
    ]


def parse_letter(text):
    m = re.search(r"\b([A-E])\b", text.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--mcqs", default=EVAL_MCQS, help="test MCQ file (causal_MCQS.json)")
    ap.add_argument("--out", default=None, help="labels json path (eid -> letter)")
    ap.add_argument("--limit", type=int, default=0, help="only first N effects (0 = all)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    effects = udiva.load_causal() if args.split == "dev" else udiva.load_mcqs(args.mcqs)
    eids = sorted(effects)
    if args.limit:
        eids = eids[: args.limit]
    print(f"{args.split}: {len(eids)} effects | model {args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    labels, n_ok = {}, 0
    for i, eid in enumerate(eids, 1):
        e = effects[eid]
        msgs = build_messages(e)
        try:  # Qwen3: disable thinking for direct deterministic answers
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        lab = parse_letter(resp) or "A"  # deterministic fallback on parse failure
        labels[eid] = lab
        if args.split == "dev" and lab in e.gt_labels:
            n_ok += 1
        if i % 25 == 0 or i == len(eids):
            acc = f" | acc {n_ok}/{i} = {n_ok/i:.4f}" if args.split == "dev" else ""
            print(f"  {i}/{len(eids)}{acc}", flush=True)

    if args.split == "dev":
        print(f"DEV accuracy: {n_ok}/{len(eids)} = {n_ok/len(eids):.4f}")
    out_path = args.out or f"{args.split}_labels.json"
    json.dump(labels, open(out_path, "w"), indent=0)
    print(f"wrote {out_path} ({len(labels)} labels)")


if __name__ == "__main__":
    main()
