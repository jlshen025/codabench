#!/usr/bin/env python
"""Assemble the evidence table for every opus-5 flip candidate.

Joins _llm/opus5_recheck.json with the 13 stored views, the 4 base-rate
recheck readers (Ecars None pool), prior_recheck (Ecars Against pool), and
the encoder/Fanar probas. Cross-family gate: a flip needs >=1 agreeing vote
from a NON-claude family (gpt5fam / deepseek / encoder / fanar).
Output: _llm/opus5_evidence.md (ranked, with AR+EN text for the eyeball pass).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
L = HERE / "_llm"

AR = pd.read_csv(HERE / "test_norm.csv", keep_default_na=False, encoding="utf-8-sig")
EN = pd.read_csv(L / "test_en.csv", keep_default_na=False, encoding="utf-8-sig")
PRED = [l.strip() for l in (HERE / "staging" / "e_v34.txt").read_text().splitlines() if l.strip()]
OPUS = {int(k): v for k, v in json.load(open(L / "opus5_recheck.json")).items()}
VIEWS = json.load(open(L / "views_all.json"))  # {view: [644 labels]}

FAMILY = {"claude": "claude", "dspro": "deepseek", "dsflash": "deepseek"}
for v in ("card2", "en", "g56", "g56fix", "g56high", "gpt5", "luna", "msa", "rag", "terra"):
    FAMILY[v] = "gpt5fam"

ID2L = {0: "Against", 1: "Favor", 2: "None"}
enc = np.load(L / "enc_test.npz")["proba"]
fnr = np.load(L / "fnr_ep2_test.npz")["proba"]

RECHECK = {}
for name in ("sol", "dspro", "terra", "luna"):
    p = L / f"none_recheck_{name}.json"
    if p.exists():
        for k, v in json.load(open(p)).items():
            RECHECK.setdefault(int(k), {})[name] = f"{v['label']}({v['conf']})"
PRIOR = {int(k): f"{v['label']}({v['conf']})"
         for k, v in json.load(open(L / "prior_recheck.json")).items()}

POOL_CUR = {"ecN": "None", "trN": "None", "ecA": "Against",
            "trA": "Against", "ecF": "Favor", "trF": "Favor"}

out = ["# opus-5 flip candidates with cross-family evidence\n"]
summary = {}
for pool, cur in POOL_CUR.items():
    rows = [(i, v) for i, v in OPUS.items() if v["pool"] == pool and v["label"] != cur]

    def support(i, lab):
        fams = {}
        for view, labels in VIEWS.items():
            if labels[i] == lab:
                fams.setdefault(FAMILY[view], []).append(view)
        if ID2L[int(np.argmax(enc[i]))] == lab:
            fams.setdefault("encoder", []).append(f"enc p={enc[i].max():.2f}")
        if ID2L[int(np.argmax(fnr[i]))] == lab:
            fams.setdefault("fanar", []).append(f"fnr p={fnr[i].max():.2f}")
        return fams

    ranked = []
    for i, v in rows:
        fams = support(i, v["label"])
        xfam = [f for f in fams if f != "claude"]
        ranked.append((len(xfam), v["conf"], i, v, fams))
    ranked.sort(reverse=True)
    summary[pool] = [(i, v["label"], len_x, v["conf"])
                     for len_x, _, i, v, _ in ranked for _ in [0]][:99]

    out.append(f"\n## {pool}  (current={POOL_CUR[pool]}, {len(rows)} opus flips)\n")
    for nx, conf, i, v, fams in ranked:
        gate = "PASS" if nx >= 1 else "FAIL"
        out.append(f"### row {i} -> {v['label']}  conf={conf}  xfam={nx} [{gate}]")
        out.append(f"  opus why: {v['why']}")
        out.append(f"  families agreeing: " +
                   ("; ".join(f"{f}: {', '.join(ws)}" for f, ws in sorted(fams.items()))
                    or "NONE (opus alone)"))
        if i in RECHECK:
            out.append(f"  base-rate rechecks: " +
                       ", ".join(f"{n}={s}" for n, s in RECHECK[i].items()))
        if i in PRIOR and pool == "ecA":
            out.append(f"  prior_recheck(sol): {PRIOR[i]}")
        out.append(f"  enc[A,F,N]: {np.round(enc[i], 2).tolist()}  "
                   f"fnr: {np.round(fnr[i], 2).tolist()}")
        out.append(f"  AR: {AR.text[i]}")
        out.append(f"  EN: {EN.text[i]}\n")

(L / "opus5_evidence.md").write_text("\n".join(out))
for pool, cands in summary.items():
    passing = [c for c in cands if c[2] >= 1]
    print(f"{pool}: {len(cands)} flips, {len(passing)} pass xfam gate: "
          + " ".join(f"{i}->{l}(x{n},c{c})" for i, l, n, c in passing))
print("wrote _llm/opus5_evidence.md")
