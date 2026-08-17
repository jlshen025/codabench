#!/usr/bin/env python
"""Merge + validate the opus-5 batch outputs into _llm/opus5_recheck.json.

Coverage contract: every row in _llm/opus5_batches/index.json appears exactly
once with a valid label. Exits nonzero listing missing batches/rows otherwise.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BDIR = HERE / "_llm" / "opus5_batches"
ODIR = HERE / "_llm" / "opus5_out"
LABELS = {"Favor", "Against", "None"}

index = json.load(open(BDIR / "index.json"))
expected = {i: pool for pool, rows in index.items() for i in rows}

merged, problems = {}, []
for f in sorted(ODIR.glob("*.json")):
    try:
        data = json.load(open(f))
    except Exception as e:  # noqa: BLE001
        problems.append(f"{f.name}: unparseable ({e})")
        continue
    for k, v in data.items():
        i = int(k)
        if i not in expected:
            problems.append(f"{f.name}: unexpected row {i}")
            continue
        if i in merged:
            problems.append(f"{f.name}: duplicate row {i}")
            continue
        lab = str(v.get("label", "")).capitalize()
        if lab not in LABELS:
            problems.append(f"{f.name}: row {i} bad label {v.get('label')!r}")
            continue
        merged[i] = {"label": lab, "conf": int(v.get("conf", 0)),
                     "why": str(v.get("why", ""))[:200], "pool": expected[i]}

missing = sorted(set(expected) - set(merged))
if missing:
    miss_batches = sorted({f"{expected[i]}" for i in missing})
    problems.append(f"missing {len(missing)} rows from pools {miss_batches}: {missing[:40]}")

json.dump(merged, open(HERE / "_llm" / "opus5_recheck.json", "w"), indent=0)

flips = {}
for i, v in merged.items():
    pool = v["pool"]
    cur = {"ecN": "None", "trN": "None", "ecA": "Against", "trA": "Against",
           "ecF": "Favor", "trF": "Favor"}[pool]
    if v["label"] != cur:
        flips.setdefault(pool, []).append((v["conf"], i, v["label"]))

print(f"merged {len(merged)}/{len(expected)} rows from {len(list(ODIR.glob('*.json')))} files")
for pool in ("ecN", "trN", "ecA", "trA", "ecF", "trF"):
    fl = sorted(flips.get(pool, []), reverse=True)
    print(f"  {pool}: {len(fl)} flips  " +
          " ".join(f"{i}->{l[0]}({c})" for c, i, l in fl))
if problems:
    print("PROBLEMS:")
    print("\n".join("  " + p for p in problems))
    sys.exit(1)
print("coverage OK")
