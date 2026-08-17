"""
thresh_tune.py — Honest nested per-class threshold tuning on the claude held-best blend.

Favg2 EXCLUDES None, so an additive offset on the class logits (esp. the None column)
before argmax can trade None<->Favor/Against to raise Favor+Against F1. RISK (Gate-2):
the optimal offset can flip sign across targets. So we tune HONESTLY: for each held-out
unseen draw, pick the offset on the OTHER two draws and apply it to the held-out one
(offset never tuned on the draw it is scored on). Adopt only if nested mean3 > baseline
AND per-draw oracle offsets are consistent.

Blend = (1-W)*enc + W*claude  (W=0.35, the held-best). Cached proba only — no relay.
"""
import glob, sys, numpy as np
sys.path.insert(0, '.')
import stance_lib as S

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35


def seedavg(prefixes, cv):
    P = []; y = None; t = None
    for pfx in prefixes:
        sp = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"results/{pfx}_{cv}_s*/oof_proba.npz"))]
        P.append(np.mean([s['proba'] for s in sp], 0)); y = sp[0]['y_true']
        if 'target' in sp[0].files:
            t = sp[0]['target'].astype(str)
    return np.mean(P, 0), y, t


ENC = ("t2_base_ara", "t2_base_marb")
enc_d, yd, _ = seedavg(ENC, "dev")
enc_l, yl, tl = seedavg(ENC, "loto")
cd = np.load("_llm/claude_devt2.npz", allow_pickle=True)['proba']
cll = np.load("_llm/claude_loto_train.npz", allow_pickle=True)['proba']
bd = (1 - W) * enc_d + W * cd
bl = (1 - W) * enc_l + W * cll
cov = tl == "Covid Vaccine"; dig = tl == "Digital Transformation"
draws = {"WomenEmp": (bd, yd), "Covid": (bl[cov], yl[cov]), "Digital": (bl[dig], yl[dig])}


def f2(p, y, off):
    return S.compute_metrics(y, (p + off).argmax(1))['Favg2'] * 100


# grid over [against, favor, none] offsets, favor fixed at 0 (only relative matters)
GRID = [np.array([ag, 0.0, no]) for ag in np.round(np.arange(-0.15, 0.151, 0.05), 3)
        for no in np.round(np.arange(-0.35, 0.351, 0.05), 3)]

print(f"=== threshold tuning, blend W={W} (claude weight) ===")
base = {k: f2(p, y, np.zeros(3)) for k, (p, y) in draws.items()}
bmean = np.mean(list(base.values()))
print("baseline (no thresh):", {k: round(v, 2) for k, v in base.items()}, "| mean3", round(bmean, 2))

print("\noracle per-draw (OPTIMISTIC — overfits that draw):")
for k, (p, y) in draws.items():
    sc = [(f2(p, y, o), o) for o in GRID]
    best = max(sc, key=lambda x: x[0])
    print(f"  {k:9s}: {best[0]:.2f}  off(ag,none)=({best[1][0]:+.2f},{best[1][2]:+.2f})  [base {base[k]:.2f}]")

print("\nnested (HONEST — offset tuned on the OTHER two draws):")
names = list(draws)
held = {}
for h in names:
    others = [n for n in names if n != h]
    best_o = max(GRID, key=lambda o: np.mean([f2(*draws[n], o) for n in others]))
    held[h] = f2(*draws[h], best_o)
    print(f"  hold {h:9s}: tuned on {others} -> off(ag,none)=({best_o[0]:+.2f},{best_o[2]:+.2f}) -> {held[h]:.2f}  [base {base[h]:.2f}]")
hmean = np.mean(list(held.values()))
print(f"\nNESTED mean3 = {hmean:.2f}   vs baseline mean3 = {bmean:.2f}   (delta {hmean - bmean:+.2f})")
print("VERDICT:", "ADOPT (robust)" if hmean > bmean + 0.05 else "RULE OUT (no robust gain / sign-flips)")
