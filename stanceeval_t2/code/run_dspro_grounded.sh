#!/bin/bash
# Run deepseek-v4-pro GROUNDED voter on the 3 unseen-target draws (login node).
# The registry is rebuilt from .env, which routes deepseek-v4-pro to the official
# https://api.deepseek.com endpoint.
set -e
cd "$(dirname "$0")"
PY=<scratch>/.venv/bin/python
DEV=<datasets>/MawqifV2/dev_track_2.csv
TRAIN=<datasets>/MawqifV2/train_track_2.csv
echo "=== [$(date +%T)] deepseek-v4-pro grounded on DEV (WomenEmp, 1400) ==="
$PY llm_grounded.py --model deepseek-v4-pro --csv "$DEV"   --order file --batch 40 --out _llm/dspro_grounded_devt2.npz
echo "=== [$(date +%T)] deepseek-v4-pro grounded on TRAIN LOTO (Covid+Digital, 2721) ==="
$PY llm_grounded.py --model deepseek-v4-pro --csv "$TRAIN" --order loto --batch 40 --out _llm/dspro_grounded_loto.npz
echo "=== [$(date +%T)] DONE ==="
