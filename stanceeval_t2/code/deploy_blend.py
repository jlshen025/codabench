"""deploy_blend.py — weighted 5-way blend of per-voter proba npz -> predictions.txt (CSV file
order) -> FLAT zip. Held-best config A weights: ENC .2 | claude .1 | gpt5 .3 | q14 .3 | fnr .1.
Every --* npz carries key 'proba' [N,3] cols [Against,Favor,None] in the SAME row order.
--gold_csv (a labeled CSV in the same order) prints Favg2/Favg3 for validation."""
import os, sys, argparse, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ORDER = ["enc", "claude", "gpt5", "q14", "fnr"]
ap = argparse.ArgumentParser()
for k in ORDER:
    ap.add_argument(f"--{k}", required=True, help=f"{k} proba npz")
ap.add_argument("--weights", default="0.2,0.1,0.3,0.3,0.1", help="enc,claude,gpt5,q14,fnr")
ap.add_argument("--out_zip", required=True)
ap.add_argument("--txt_name", default="predictions.txt")
ap.add_argument("--gold_csv", default="")
a = ap.parse_args()


def load(p):
    d = np.load(p, allow_pickle=True)
    return d["proba"].astype(np.float64)


arrs = {k: load(getattr(a, k)) for k in ORDER}
n = len(arrs["enc"])
for k, v in arrs.items():
    assert len(v) == n, f"{k} len {len(v)} != {n}"
    assert v.shape[1] == 3, f"{k} shape {v.shape}"
w = [float(x) for x in a.weights.split(",")]
assert len(w) == 5 and abs(sum(w) - 1.0) < 1e-6, f"weights {w} sum {sum(w)}"
blend = sum(w[i] * arrs[ORDER[i]] for i in range(5))
pred = blend.argmax(1)
labels = [S.ID2LABEL[i] for i in pred]
os.makedirs(os.path.dirname(os.path.abspath(a.out_zip)), exist_ok=True)
txt = os.path.join(os.path.dirname(os.path.abspath(a.out_zip)), a.txt_name)
open(txt, "w").write("\n".join(labels) + "\n")
with zipfile.ZipFile(a.out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(txt, a.txt_name)
dist = {S.ID2LABEL[k]: int((pred == k).sum()) for k in range(3)}
print(f"[blend] w(enc,cl,g5,q14,fnr)={w} n={n} -> {a.out_zip} (FLAT) dist={dist}", flush=True)
if a.gold_csv:
    g = pd.read_csv(a.gold_csv, keep_default_na=False, dtype=str)
    assert len(g) == n, f"gold len {len(g)} != {n}"
    y = np.array([S.LABEL2ID[s.strip()] for s in g["stance"]])
    m = S.compute_metrics(y, pred)
    print(f"[VAL blend] Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f}", flush=True)
