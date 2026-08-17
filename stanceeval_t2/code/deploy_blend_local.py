"""deploy_blend_local.py — POST-FENCE local blend {enc,q14,fnr[,cand]} -> predictions txt (CSV row
order) -> FLAT zip. No claude/gpt5 voters (dropped by design). Each --* npz carries key
'proba' [N,3] cols [Against,Favor,None] in the SAME row order. --gold_csv (labeled, same order)
prints Favg2/Favg3 for validation."""
import os, sys, argparse, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True)
ap.add_argument("--q14", required=True)
ap.add_argument("--fnr", required=True)
ap.add_argument("--cand", default="", help="optional 4th voter proba npz (fresh relay grounded)")
ap.add_argument("--weights", required=True, help="enc,q14,fnr[,cand] comma-sep, sum=1")
ap.add_argument("--out_zip", required=True)
ap.add_argument("--txt_name", default="predictions.txt")
ap.add_argument("--gold_csv", default="")
a = ap.parse_args()


def load(p):
    return np.load(p, allow_pickle=True)["proba"].astype(np.float64)


names = ["enc", "q14", "fnr"] + (["cand"] if a.cand else [])
paths = {"enc": a.enc, "q14": a.q14, "fnr": a.fnr}
if a.cand:
    paths["cand"] = a.cand
arrs = {k: load(paths[k]) for k in names}
n = len(arrs["enc"])
for k, v in arrs.items():
    assert len(v) == n, f"{k} len {len(v)} != {n}"
    assert v.shape[1] == 3, f"{k} shape {v.shape}"
w = [float(x) for x in a.weights.split(",")]
assert len(w) == len(names), f"weights {w} count != voters {names}"
assert abs(sum(w) - 1.0) < 1e-6, f"weights sum {sum(w)} != 1"
blend = sum(w[i] * arrs[names[i]] for i in range(len(names)))
pred = blend.argmax(1)
labels = [S.ID2LABEL[i] for i in pred]
os.makedirs(os.path.dirname(os.path.abspath(a.out_zip)), exist_ok=True)
txt = os.path.join(os.path.dirname(os.path.abspath(a.out_zip)), a.txt_name)
open(txt, "w").write("\n".join(labels) + "\n")
with zipfile.ZipFile(a.out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(txt, a.txt_name)
dist = {S.ID2LABEL[k]: int((pred == k).sum()) for k in range(3)}
print(f"[blend-local] names={names} w={w} n={n} -> {a.out_zip} (FLAT) dist={dist}", flush=True)
if a.gold_csv:
    g = pd.read_csv(a.gold_csv, keep_default_na=False, dtype=str)
    assert len(g) == n, f"gold len {len(g)} != {n}"
    y = np.array([S.LABEL2ID[s.strip()] for s in g["stance"]])
    m = S.compute_metrics(y, pred)
    print(f"[VAL blend-local] Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f}", flush=True)
