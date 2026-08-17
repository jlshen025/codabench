# UDIVA-HHOI Track 5 — Multimodal Exocentric Causal Event Grounding

Team **JLShen** (Codabench user `junlong`) · Competition 16647 · Test phase 27275
Final submission: **835537** — server temporal accuracy **0.4274** / MC accuracy **0.7564**.

Fully local pipeline (see the fact sheet for the complete method description and model
disclosure).

## Method

Two independent components, matching the two independent metric columns:

1. **Cause selection (MC accuracy).** Each effect is posed as a zero-shot single-choice
   text question to an instruction-tuned LLM: the effect description plus the five
   candidate-cause options (A-E) from the challenge MCQ files, under a system prompt
   framing the dyadic Lego-assembly setting; deterministic decoding; the answer is one
   letter. No fine-tuning; no video/audio input.
2. **Timestamp (temporal accuracy).** A deterministic constant offset,
   `t̂ = t_b(effect) − 1.15 s`, calibrated on the development set. Reproduces exactly.

## Run (fully local, no network needed after the one-time model download)

```bash
pip install -r requirements.txt        # torch, transformers>=4.51, accelerate

# development set (300 effects, 21 annotated sessions): prints accuracy vs the reference
python code/qwen_mcq_local.py --split dev

# test set: produce labels, then build the submission zip
python code/qwen_mcq_local.py --split test --out test_labels.json
python code/make_submission.py test_labels.json 1.15 causal_submission.zip
```

Default model: **Qwen3-32B** (bf16, one 80 GB GPU, `device_map="auto"`);
`--model Qwen/Qwen3-8B` fits a 24 GB GPU; a tiny model (e.g.
`Qwen/Qwen2.5-0.5B-Instruct`) with `--limit` gives a fast CPU smoke test.
Greedy decoding, thinking mode disabled — deterministic.

Adjust the two dataset path constants at the top of `code/udiva.py` and
`code/qwen_mcq_local.py` to your UDIVA-HHOI location.

## Pipeline map

| step | script |
|---|---|
| data loading (dev reference / test MCQs) | `code/udiva.py` |
| local LLM inference (dev accuracy / test labels) | `code/qwen_mcq_local.py` |
| submission builder (labels → causal.json zip) | `code/make_submission.py` |

## License

MIT (see `LICENSE`); the code remains publicly accessible for at least three years.
