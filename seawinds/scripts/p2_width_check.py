"""Acceptance check for a p2_width_probe.py rescale: prove it moved EXACTLY what it claims.

The probe edits q05/q95 at horizons 7 and 14 only. That is a narrow, checkable contract, and
checking it is cheap next to the cost of spending one of ten final-window submissions on a
zip that silently also moved d1, a direction column or the row order.

Asserts, against the SOURCE csv:
  * every column except q05/q95 is byte-identical (that includes q50, all three direction
    columns, and every metadata/ordering column);
  * q05/q95 differ on ~all d7+d14 rows and on NO d1 row (checked by re-deriving the count);
  * the realised mean interval width ratio per horizon equals the requested scale ratio.

Run it with --scale7 1.10 --scale14 1.10 for the NO-OP control: same code path, expected
result "identical". A rescale tool that cannot return the input unchanged is not trustworthy
when it does change something.

  python p2_width_check.py --src SRC.csv --cand CAND.csv --scale7 1.03 --scale14 0.90
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

SHIPPED_SCALE = 1.10
META = ["type", "window", "region", "latitude", "longitude", "horizon", "hour", "level"]
KEEP = META + ["q50", "dir_05", "dir_50", "dir_95"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--scale7", type=float, required=True)
    ap.add_argument("--scale14", type=float, required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import p2_predcsv as PC

    src = PC.read_predictions(a.src)
    cand = PC.read_predictions(a.cand)
    rep = PC.compare(src, cand)

    fails = []
    for c in KEEP:
        if rep[c]["n_diff"] != 0:
            fails.append(f"{c}: {rep[c]['n_diff']:,} rows moved (max_abs {rep[c]['max_abs']}) "
                         f"— this column must be untouched")

    h = src["horizon"].to_numpy()
    n_d7d14 = int(((h == 7) | (h == 14)).sum())
    for c in ("q05", "q95"):
        n = rep[c]["n_diff"]
        want = {7: a.scale7, 14: a.scale14}
        noop = all(abs(v - SHIPPED_SCALE) < 1e-9 for v in want.values())
        if noop:
            if n != 0:
                fails.append(f"NO-OP CONTROL FAILED: {c} moved on {n:,} rows at scale 1.10")
        elif not (0.98 * n_d7d14 <= n <= n_d7d14):
            fails.append(f"{c}: {n:,} rows moved, expected ~{n_d7d14:,} (d7+d14 only)")

    # per-horizon realised width ratio must equal the requested scale ratio
    print(f"{'h':>4} {'src width':>11} {'cand width':>11} {'ratio':>8} {'expected':>9}  ok")
    for hh, target in ((1, None), (7, a.scale7), (14, a.scale14)):
        m = (h == hh)
        ws = float((src.loc[m, "q95"] - src.loc[m, "q05"]).mean())
        wc = float((cand.loc[m, "q95"] - cand.loc[m, "q05"]).mean())
        r = wc / ws if ws else float("nan")
        exp = 1.0 if target is None else target / SHIPPED_SCALE
        ok = abs(r - exp) < 2e-3
        print(f"{hh:>4} {ws:>11.5f} {wc:>11.5f} {r:>8.5f} {exp:>9.5f}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"d{hh}: realised width ratio {r:.5f} != requested {exp:.5f}")

    # quantile sanity on the candidate (the scorer rejects violations)
    bad = int((cand["q05"] > cand["q50"] + 1e-9).sum() + (cand["q50"] > cand["q95"] + 1e-9).sum())
    if bad:
        fails.append(f"{bad:,} rows violate q05<=q50<=q95")
    if int((cand["q05"] < 0).sum()):
        fails.append("negative q05 present")

    print(f"\nrows {len(cand):,}   d7+d14 {n_d7d14:,}   "
          f"q05 moved {rep['q05']['n_diff']:,}   q95 moved {rep['q95']['n_diff']:,}")
    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("\nACCEPTANCE PASSED — only q05/q95 at d7/d14 moved, by exactly the requested ratios")


if __name__ == "__main__":
    main()
