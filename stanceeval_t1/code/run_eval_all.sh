#!/bin/bash
# run_eval_all.sh — produce ALL Track-1 eval candidates from ONE Qwen inference.
# Track-1 blind test = UNSEEN target "Women Driving" → dev-CV is NOT the proxy; compare
# Qwen-heavy vs encoder blends on the SERVER. Candidates (all share the same Qwen prompt-ens):
#   ce.zip   = CE-enc(0.15)  ⊕ Qwen(0.85)   [held-best recipe]
#   sc.zip   = SupCon-enc(0.15) ⊕ Qwen(0.85)[robust encoder — may transfer better to unseen]
#   qwen.zip = Qwen-ens alone (w_llm=1.0)    [no encoder — hypothesis: best for unseen]
#   ce05.zip = CE-enc(0.05) ⊕ Qwen(0.95)     [encoder-light]
# Usage: run_eval_all.sh <TEST_CSV> <OUT_DIR>
set -euo pipefail
TEST_CSV="$1"; OUT_DIR="$2"
SD="${SD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
R="$SD/results"; LL="$OUT_DIR/_llm"; BASE="Qwen/Qwen2.5-14B-Instruct"
mkdir -p "$LL" "$OUT_DIR"
CE_ENC=("$R/deploy_enc_mb_s42/model" "$R/deploy_enc_atw_s42/model")
SC_ENC=("$R/deploy_enc_sc_mb_s42/model" "$R/deploy_enc_sc_mb_s1/model" "$R/deploy_enc_sc_atw_s42/model" "$R/deploy_enc_sc_atw_s1/model")

echo "=== [1] Qwen predict_lora ar_letter (seed-avg s42,s1) ==="
python "$SD/predict_lora.py" --base_model "$BASE" \
  --adapters "$R/deploy_q14ar_s42/adapter" "$R/deploy_q14ar_s1/adapter" \
  --verbalizer ar_letter --test_csv "$TEST_CSV" --out_npz "$LL/dep_ar.npz"
echo "=== [2] Qwen predict_lora lat_letter (seed-avg s42,s1) ==="
python "$SD/predict_lora.py" --base_model "$BASE" \
  --adapters "$R/deploy_q14lat_s42/adapter" "$R/deploy_q14lat_s1/adapter" \
  --verbalizer lat_letter --test_csv "$TEST_CSV" --out_npz "$LL/dep_lat.npz"
echo "=== [3] avg prompts -> Qwen ensemble ==="
python - "$LL/dep_ar.npz" "$LL/dep_lat.npz" "$LL/dep_qens.npz" <<'PY'
import sys, numpy as np
ar=np.load(sys.argv[1])["proba"].astype(float); lat=np.load(sys.argv[2])["proba"].astype(float)
ens=np.mean([ar,lat],0).astype(np.float32); np.savez(sys.argv[3], proba=ens)
print(f"[qens] {ens.shape} dist={np.bincount(ens.argmax(1),minlength=3).tolist()}")
PY

blend () {  # <name> <w_llm> <enc dirs...>
  local name="$1" w="$2"; shift 2
  python "$SD/predict_test.py" --models "$@" --test_csv "$TEST_CSV" \
    --llm_npz "$LL/dep_qens.npz" --w_llm "$w" --preprocess baseline --out_zip "$OUT_DIR/$name.zip"
  python - "$TEST_CSV" "$OUT_DIR/$name.zip" "$name" <<'PY'
import sys, pandas as pd, zipfile
from collections import Counter
n=len(pd.read_csv(sys.argv[1],keep_default_na=False,dtype=str))
z=zipfile.ZipFile(sys.argv[2]); nm=[x for x in z.namelist() if x.endswith(".txt")][0]
pr=z.read(nm).decode().splitlines()
assert n==len(pr) and set(pr)<={"Favor","Against","None"} and "/" not in nm, f"BAD {sys.argv[3]}"
print(f"  {sys.argv[3]}: rows={n} FLAT ok dist={dict(Counter(pr))}")
PY
}
echo "=== [4] candidates ==="
blend ce   0.85 "${CE_ENC[@]}"
blend sc   0.85 "${SC_ENC[@]}"
blend qwen 1.0  "${CE_ENC[@]}"
blend ce05 0.95 "${CE_ENC[@]}"
echo "=== DONE: $OUT_DIR/{ce,sc,qwen,ce05}.zip ==="
ls -la "$OUT_DIR"/*.zip