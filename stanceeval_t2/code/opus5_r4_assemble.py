#!/usr/bin/env python
"""Assemble ROUND 4 shortlist: convention-simulation flips joined with every
prior signal (views, sol prior_recheck, R1 opus recheck, xchecks, oss/gemini).
Prints per-pool ranked candidates with all evidence + text for the eyeball gate.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
L = HERE / "_llm"
ar = pd.read_csv(HERE / "test_norm.csv", keep_default_na=False, encoding="utf-8-sig")
en = pd.read_csv(L / "test_en.csv", keep_default_na=False, encoding="utf-8-sig")
views = json.load(open(L / "views_all.json"))
opus1 = {int(k): v for k, v in json.load(open(L / "opus5_recheck.json")).items()}
prior = {int(k): v for k, v in json.load(open(L / "prior_recheck.json")).items()}

conv = {}
for f in sorted((L / "opus5_conv_out").glob("*.json")):
    pool = f.stem.rsplit("_", 1)[0]
    for k, v in json.load(open(f)).items():
        conv[int(k)] = {**v, "pool": pool}

CUR = {"ecAF": "Against", "trFA": "Favor", "ecNF": "None"}
extra = {}
for name in ("xsol_ecA", "xdspro_ecA", "xoss_ecA", "xgem_ecA", "xoss_r2"):
    p = L / f"opus5_{name}.json"
    if p.exists():
        for k, v in json.load(open(p)).items():
            extra.setdefault(int(k), []).append(f"{name.split('_')[0]}:{v['label']}({v['conf']})")

for pool, cur in CUR.items():
    cands = [(i, v) for i, v in conv.items() if v["pool"] == pool
             and v["label"].capitalize() != cur]
    ranked = []
    for i, v in cands:
        want = v["label"].capitalize()
        vv = sum(1 for lab in views.values() if lab[i] == want)
        o1 = opus1.get(i, {})
        pr = prior.get(i, {})
        score = (vv >= 1) + (o1.get("label") == want) + (pr.get("label") == want)
        ranked.append((score, int(v.get("conf", 0)), i, v, vv, o1, pr))
    ranked.sort(reverse=True)
    print(f"\n### {pool} (cur={cur}): {len(cands)} conv flips")
    for score, conf, i, v, vv, o1, pr in ranked:
        line = (f"{i} ->{v['label']}({conf}) support={score} views={vv} "
                f"opusR1={o1.get('label')}({o1.get('conf')})")
        if pr:
            line += f" solPrior={pr.get('label')}({pr.get('conf')})"
        if i in extra:
            line += " | " + " ".join(extra[i])
        print(line)
        print(f"   why: {v.get('why','')[:110]}")
        print(f"   EN: {' '.join(en.text[i].split())[:150]}")
