#!/usr/bin/env python
"""Apply a pre-registered flip set to the champion vector and build the zip.

Usage: [OPUS5_BASE=e_o5final] opus5_apply.py <name> <idx:Label> [<idx:Label> ...]
Writes staging/e_<name>.txt + staging/sub_<name>.zip (flat, predictions.txt).
Refuses: <4 flips (Gate-1 no-small-diffs), a flip equal to the current label,
or any row in the v34 re-adjudication fence.
"""
import os
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "staging" / (os.environ.get("OPUS5_BASE", "e_v34") + ".txt")
LABELS = {"Favor", "Against", "None"}
FENCE = {320, 345, 360, 363, 391, 394, 398, 424, 437, 441, 464, 480, 492,
         495, 505, 534, 542, 551, 570, 574, 581,
         56, 66, 92, 141, 144, 158, 173, 211, 219, 238, 242}

name = sys.argv[1]
flips = {}
for spec in sys.argv[2:]:
    k, lab = spec.split(":")
    assert lab in LABELS, f"bad label {lab}"
    flips[int(k)] = lab

pred = [l.strip() for l in BASE.read_text().splitlines() if l.strip()]
assert len(pred) == 644
assert len(flips) >= 4, f"only {len(flips)} flips — Gate-1 requires >=4-row diffs"
bad_fence = sorted(set(flips) & FENCE)
assert not bad_fence, f"rows {bad_fence} are v34 flip rows (re-adjudication fence)"
for i, lab in flips.items():
    assert pred[i] != lab, f"row {i} already {lab}"
    pred[i] = lab

txt = HERE / "staging" / f"e_{name}.txt"
txt.write_text("\n".join(pred) + "\n")
zp = HERE / "staging" / f"sub_{name}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("predictions.txt", txt.read_text())

base = [l.strip() for l in BASE.read_text().splitlines() if l.strip()]
diff = [(i, base[i], pred[i]) for i in range(644) if base[i] != pred[i]]
assert sorted(i for i, _, _ in diff) == sorted(flips), "diff != flip set"
print(f"{name}: {len(diff)} flips vs v34 -> {zp.name}")
for i, a, b in diff:
    print(f"  {i}: {a}->{b}")
