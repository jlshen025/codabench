"""Exhaustive K=5 search over a curated pool; report top sets by mean4 with per-column
scores so I can pick a BALANCED one (official metric = rank-avg over 4 columns, so don't
tank any column). Alts rebuilt per fold from train mining (only composition is tuned)."""
import itertools
from collections import Counter
import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P
import tune2

PA, PB = "participant_a", "participant_b"


def curated_pool(m):
    """name->seq, ~12 items, built from mined train freqs m."""
    sA = [P.k2e(list(k)) for k, _ in m["singAll"].most_common(6)]
    sV = [P.k2e(list(k)) for k, _ in m["singV"].most_common(4)]
    sNV = [P.k2e(list(k)) for k, _ in m["singNV"].most_common(6)]
    seqAll = [tune2.sk2seq(sk) for sk, _ in m["seqAll"].most_common(8)]
    seqNV = [tune2.sk2seq(sk) for sk, _ in m["seqNV"].most_common(6)]
    # pick multi-event sequences for 'full'
    full_multi = [s for s in seqAll if len(s) >= 2][:2]
    nv_multi = [s for s in seqNV if len(s) >= 2][:2]
    pool = {"empty": []}
    for i, s in enumerate(sNV[:4]):
        pool[f"NV{i+1}"] = s
    for i, s in enumerate(sV[:2]):
        pool[f"V{i+1}"] = s
    for i, s in enumerate(full_multi):
        pool[f"F{i+1}"] = s
    for i, s in enumerate(nv_multi):
        pool[f"NVm{i+1}"] = s
    # dedup
    seen, out = set(), {}
    for n, s in pool.items():
        key = tuple(tuple(sdl.ev_key(e)) for e in s)
        if key in seen:
            continue
        seen.add(key)
        out[n] = s
    return out


def make_factory(names):
    def fac(train, gt):
        pool = curated_pool(tune2.mine(train, gt))
        alts = [pool[n] for n in names if n in pool]
        return tune2.PoolHedge(alts)
    return fac


if __name__ == "__main__":
    gt = CV.build_all_gt()
    names = list(curated_pool(tune2.mine(list(gt.keys()), gt)).keys())
    print("pool (%d):" % len(names), names)
    results = []
    combos = list(itertools.combinations(names, 5))
    print("evaluating %d combos..." % len(combos))
    for i, combo in enumerate(combos):
        r = CV.evaluate(make_factory(list(combo)), gt)
        results.append((r["mean4"], r["next"], r["verbal"], r["nonverbal"], r["full"], combo))
        if i % 100 == 0:
            print("  ..%d/%d" % (i, len(combos)), flush=True)
    results.sort(reverse=True)
    print("\nTOP 15 by mean4 [mean4 next verb nonv full] combo:")
    for m4, nx, vb, nv, fl, combo in results[:15]:
        bal = "BAL" if (vb > 0.54 and fl > 0.30 and nv > 0.40 and nx > 0.38) else "   "
        print("  %.4f  n=%.3f v=%.3f nv=%.3f f=%.3f  %s  %s" % (m4, nx, vb, nv, fl, bal, combo))
    # best balanced
    balanced = [r for r in results if r[2] > 0.54 and r[4] > 0.30 and r[3] > 0.40 and r[1] > 0.38]
    if balanced:
        print("\nBEST BALANCED:", "%.4f" % balanced[0][0], balanced[0][5])
