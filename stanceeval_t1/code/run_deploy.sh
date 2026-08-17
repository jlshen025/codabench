#!/bin/bash
# run_deploy.sh — full Track-1 deploy pipeline (held-best 89.78 architecture).
#   enc(0.15) + Qwen-14B-LoRA prompt-ensemble(0.85), all OFFLINE (shared /project cache).
# Usage: run_deploy.sh <TEST_CSV> <OUT_ZIP>
# Adapters/encoders = the DEPLOY models retrained on train+dev (results/deploy_*).
set -euo pipefail
TEST_CSV="$1"; OUT_ZIP="$2"
SD="${SD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
R="$SD/results"; LL="$SD/_llm"; BASE="Qwen/Qwen2.5-14B-Instruct"
mkdir -p "$LL" "$(dirname "$OUT_ZIP")"

echo "=== [1/4] predict_lora ar_letter (seed-avg s42,s1) ==="
python "$SD/predict_lora.py" --base_model "$BASE" \
  --adapters "$R/deploy_q14ar_s42/adapter" "$R/deploy_q14ar_s1/adapter" \
  --verbalizer ar_letter --test_csv "$TEST_CSV" --out_npz "$LL/dep_ar.npz"

echo "=== [2/4] predict_lora lat_letter (seed-avg s42,s1) ==="
python "$SD/predict_lora.py" --base_model "$BASE" \
  --adapters "$R/deploy_q14lat_s42/adapter" "$R/deploy_q14lat_s1/adapter" \
  --verbalizer lat_letter --test_csv "$TEST_CSV" --out_npz "$LL/dep_lat.npz"

echo "=== [3/4] average the 2 prompts -> Qwen prompt-ensemble ==="
python - "$LL/dep_ar.npz" "$LL/dep_lat.npz" "$LL/dep_qens.npz" <<'PY'
import sys, numpy as np
ar=np.load(sys.argv[1])["proba"].astype(float); lat=np.load(sys.argv[2])["proba"].astype(float)
ens=np.mean([ar,lat],0).astype(np.float32)
np.savez(sys.argv[3], proba=ens)
print(f"[avg] ar{ar.shape}+lat{lat.shape} -> qens{ens.shape} dist={np.bincount(ens.argmax(1),minlength=3).tolist()}")
PY

echo "=== [4/4] predict_test blend enc(0.15)+qens(0.85) -> FLAT zip ==="
python "$SD/predict_test.py" \
  --models "$R/deploy_enc_mb_s42/model" "$R/deploy_enc_atw_s42/model" \
  --test_csv "$TEST_CSV" --llm_npz "$LL/dep_qens.npz" --w_llm 0.85 \
  --preprocess baseline --out_zip "$OUT_ZIP"

echo "=== zip contents (must be FLAT: predictions.txt at root) ==="
unzip -l "$OUT_ZIP"
echo "=== ROBUST sanity: predictions == test rows, FLAT, valid labels ==="
python - "$TEST_CSV" "$OUT_ZIP" <<'PY'
import sys, pandas as pd, zipfile
from collections import Counter
n_test = len(pd.read_csv(sys.argv[1], keep_default_na=False, dtype=str))
z = zipfile.ZipFile(sys.argv[2]); nm = [x for x in z.namelist() if x.endswith(".txt")][0]
preds = z.read(nm).decode().splitlines()
flat = "/" not in nm
labels_ok = set(preds) <= {"Favor", "Against", "None"}
print(f"test_rows={n_test} predictions={len(preds)} MATCH={n_test==len(preds)} flat={flat} labels_ok={labels_ok}")
print("label dist:", dict(Counter(preds)))
assert n_test == len(preds), f"COUNT MISMATCH {n_test} vs {len(preds)}"
assert labels_ok, f"BAD LABEL(S): {set(preds) - {'Favor','Against','None'}}"
assert flat, f"NOT FLAT: {nm}"
print("PIPELINE CHECK: PASS")
PY
echo "=== DONE deploy -> $OUT_ZIP ==="
