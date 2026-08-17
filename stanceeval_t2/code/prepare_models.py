"""
prepare_models.py — Stage Arabic encoders as offline, safetensors-only local dirs.

Why: transformers 5.9 refuses to load .bin (pickle) weights unless torch>=2.6
(we have 2.5.1). MARBERTv2 / CAMeLBERT ship only .bin. We load the .bin with raw
torch.load(weights_only=True) (which works in 2.5.1), drop the MLM head, and save
model.safetensors into a clean local dir so SLURM jobs load fully offline by path.

Run once on a login node (needs internet). Output: ~/scratch/stanceeval_shared/models/<short>/
"""
import os, torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from safetensors.torch import save_file

OUT_ROOT = ("<scratch>/models")
# short_name -> hub repo. Models that already ship safetensors are copied as-is.
BIN_MODELS = {
    "marbertv2": "UBC-NLP/MARBERTv2",
    "camelbert-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
}

os.makedirs(OUT_ROOT, exist_ok=True)
for short, repo in BIN_MODELS.items():
    outdir = os.path.join(OUT_ROOT, short)
    if os.path.exists(os.path.join(outdir, "model.safetensors")):
        print(f"SKIP {short} (already staged)")
        continue
    os.makedirs(outdir, exist_ok=True)
    local = snapshot_download(repo, allow_patterns=[
        "*.json", "*.txt", "*.bin", "*.model", "vocab*", "tokenizer*",
        "special_tokens*", "merges*", "sentencepiece*", "spiece*",
    ])
    AutoTokenizer.from_pretrained(local).save_pretrained(outdir)
    AutoConfig.from_pretrained(local).save_pretrained(outdir)
    binp = os.path.join(local, "pytorch_model.bin")
    sd = torch.load(binp, map_location="cpu", weights_only=True)
    # keep encoder weights, drop MLM head (cls.*); fresh classifier head trains downstream
    enc = {k: v.contiguous().clone() for k, v in sd.items() if not k.startswith("cls.")}
    save_file(enc, os.path.join(outdir, "model.safetensors"))
    m = AutoModelForSequenceClassification.from_pretrained(outdir, num_labels=3)
    print(f"OK  {short:14s} <- {repo:40s} params={sum(p.numel() for p in m.parameters())/1e6:6.1f}M")
    del m
print("done. staged dirs under", OUT_ROOT)
