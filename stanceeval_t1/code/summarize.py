"""summarize.py — scan results/*/metrics.json, tabulate Favg2 (mean±std over seeds) per config."""
import os, json, glob, collections, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = []
for mp in glob.glob(os.path.join(ROOT, "*", "metrics.json")):
    try:
        d = json.load(open(mp))
        if "pooled_Favg2" not in d:
            continue  # skip cv=full final-model metrics (no CV score)
        d["_name"] = os.path.basename(os.path.dirname(mp)); rows.append(d)
    except Exception:
        pass


def tag(d):  # prefix before first underscore of the run-dir name
    return d["_name"].split("_")[0]


def key(d):
    lam = f"s{d.get('lam_sent')}/k{d.get('lam_sarc')}" if d.get("mtl") else "-"
    return (tag(d), d.get("model"), d.get("cv"), bool(d.get("class_weights")), bool(d.get("mtl")),
            lam, d.get("preprocess", "baseline"), d.get("epochs"))


groups = collections.defaultdict(list)
for d in rows:
    groups[key(d)].append(d)

print(f"{'tag':5s} {'model':16s} {'cv':5s} {'cw':2s} {'mtl':3s} {'lam':10s} {'prep':9s} {'ep':2s} | {'Favg2 mean±std(n)':20s} {'Favg3':6s} seeds")
print("-" * 104)
for k, ds in sorted(groups.items(), key=lambda kv: (kv[0][2], -np.mean([d["pooled_Favg2"] for d in kv[1]]))):
    tg, model, cv, cw, mtl, lam, prep, ep = k
    f2 = np.array([d["pooled_Favg2"] for d in ds], float)
    f3 = np.mean([d["pooled_Favg3"] for d in ds])
    s = f2.std() if len(f2) > 1 else 0.0
    seeds = sorted(d.get("seed") for d in ds)
    print(f"{tg:5s} {str(model):16s} {cv:5s} {('Y' if cw else '-'):2s} {('Y' if mtl else '-'):3s} {lam:10s} {prep:9s} "
          f"{str(ep):2s} | {f2.mean():6.2f} ± {s:4.2f} (n={len(f2)})   {f3:6.2f} {seeds}")
