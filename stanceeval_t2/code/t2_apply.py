#!/usr/bin/env python
"""Apply a flip set to the CHAMPION vector (869013) and build a flat submission zip.

Usage: t2_apply.py <name> <idx:Label> [<idx:Label> ...]
       t2_apply.py <name> --from-file flips.txt      # lines "idx:Label"
Writes staging/e_<name>.txt + staging/sub_<name>.zip (flat, predictions.txt at root).

GATE-1 FENCE (non-negotiable, see decode_conf.py):
  * DECODE-PINNED rows whose gold label was pinned by score arithmetic rather than by
    reading the text are permanently unharvestable — flipping them would be
    membership-from-decode, the optimizer-over-assignments the competition rules forbid.
  * Every diff must be >=4 rows, so no submission isolates a single row's label.
Row membership must come from judgment about the TEXT (a reader/model/rule) only.
"""
import os, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "staging" / (os.environ.get("T2_BASE", "e_ch869013") + ".txt")
LABELS = {"Favor", "Against", "None"}

# Rows whose gold label was PINNED BY DECODE (not by reading) — permanently off-limits,
# plus the one-right-one-wrong pairs that must stay whole (splitting them would resolve
# an identity that only score arithmetic knows).
DECODE_PINNED = {575, 138, 510, 428}
PINNED_PAIRS = [{409, 456}, {124, 206}, {498, 501}, {525, 332}]

name = sys.argv[1]
args = sys.argv[2:]
if args and args[0] == "--from-file":
    args = [l.strip() for l in open(args[1]) if l.strip() and not l.startswith("#")]

flips = {}
for spec in args:
    k, lab = spec.split(":")
    assert lab in LABELS, f"bad label {lab}"
    flips[int(k)] = lab

pred = [l.strip() for l in BASE.read_text().splitlines() if l.strip()]
assert len(pred) == 644, f"base has {len(pred)} rows"
assert len(flips) >= 4, f"only {len(flips)} flips — Gate-1 requires >=4-row diffs"

bad = sorted(set(flips) & DECODE_PINNED)
assert not bad, f"GATE-1: rows {bad} are decode-pinned — membership-from-decode is forbidden"
for pair in PINNED_PAIRS:
    hit = pair & set(flips)
    assert len(hit) != 1, f"GATE-1: pair {sorted(pair)} must stay whole; you touched {sorted(hit)}"

for i, lab in flips.items():
    assert pred[i] != lab, f"row {i} is already {lab}"
    pred[i] = lab

txt = HERE / "staging" / f"e_{name}.txt"
txt.write_text("\n".join(pred) + "\n")
zp = HERE / "staging" / f"sub_{name}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("predictions.txt", txt.read_text())

base = [l.strip() for l in BASE.read_text().splitlines() if l.strip()]
diff = [(i, base[i], pred[i]) for i in range(644) if base[i] != pred[i]]
assert sorted(i for i, _, _ in diff) == sorted(flips), "diff != flip set"
print(f"{name}: {len(diff)} flips vs {BASE.stem} -> {zp}")
for i, a, b in diff:
    print(f"  {i}: {a}->{b}")
