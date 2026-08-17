"""enc_to_npz.py — encoder soft-vote proba -> npz (CSV file order), for the 5-way test deploy.
Reuses predict_test.encoder_proba (same models_final_v2 deploy encoders). Columns [Against,Favor,None].
If the CSV has a filled 'stance' column, prints Favg2 (dev validation)."""
import os, sys, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from predict_test import encoder_proba  # NB: sets HF_HOME=/project cache + offline at import

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, help="saved encoder dirs (models_final_v2/*_base)")
ap.add_argument("--csv", required=True)
ap.add_argument("--preprocess", default="baseline")
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--out_npz", required=True)
a = ap.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(a.csv, keep_default_na=False, dtype=str)
for c in ("text", "target"):
    df[c] = df[c].astype(str).str.strip()
text = df["text"].apply(S.PREPROCESSORS[a.preprocess]).tolist()
tgt = df["target"].tolist()
enc = np.mean([encoder_proba(m, tgt, text, a.max_len, device) for m in a.models], axis=0)
os.makedirs(os.path.dirname(os.path.abspath(a.out_npz)), exist_ok=True)
np.savez(a.out_npz, proba=enc.astype(np.float32))
print(f"[enc] {len(a.models)} model(s) {enc.shape} -> {a.out_npz} dist={np.bincount(enc.argmax(1),minlength=3).tolist()}", flush=True)
if "stance" in df.columns and (df["stance"].astype(str).str.strip() != "").all():
    y = np.array([S.LABEL2ID[s.strip()] for s in df["stance"]])
    m = S.compute_metrics(y, enc.argmax(1))
    print(f"[VAL enc] Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f}", flush=True)
