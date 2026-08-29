"""Predict the 2022 speed-PI width optimum by COVERAGE TARGETING, positive-controlled at d1.

WHY. Four independent server reads now agree that the speed-Winkler optimum sits at a
roughly stable EMPIRICAL COVERAGE (~88-91%), while the SCALE needed to reach it moves
year to year:
    2021 d1  1.00 -> cov 93.6 (wide)   0.90 -> cov 90.5  OPTIMUM   (A36, -0.20)
    2021 d7  1.10 -> cov 88.3  OPTIMUM; 1.00 scored +0.72 WORSE    (A35)
    2021 d14 1.10 -> cov 90.7  OPTIMUM                             (A35)
    2022 d1  0.90 -> cov 87.1 (narrow)  1.00 -> cov 90.7 OPTIMUM   (A40, -0.049)
That reconciles the standing CV-vs-server contradiction: CV is not wrong about the curve's
SHAPE, only about the LEVEL, because CV over-covers an unseen year. Coverage is the
observable that transfers.

CONSEQUENCE, and the point of this script. On 2022 the shipped d7/d14 scale 1.10 measured
coverage 92.8 / 96.9 -- both ABOVE the optimal band, d14 grossly so. So d7/d14 are
over-wide on the deciding year and should be NARROWED, which is the opposite of the 2021
verdict and could not have been guessed from either CV or the 2021 read alone.

METHOD. Take the SHAPE (scale -> coverage, winkler) from the leave-year-out CV bracket
(A33), and the LEVEL from the server. For horizon h:
  1. find the CV scale s_eq whose CV coverage equals the server-observed coverage at the
     shipped scale -- rho = s_eq / s_ship is the year's error-inflation vs CV;
  2. the CV-optimal scale s_cv* maps to the real scale s* = s_cv* / rho;
  3. predict the gain by reading the CV winkler curve between s_eq and s_cv*, rescaled to
     the server's observed level.

POSITIVE CONTROL (the reason to believe any of this). d1 has TWO measured 2022 points.
Fit the model on the 0.90 point ALONE and predict the 1.00 point. If the prediction misses
the measured (8.9825, cov 90.7), the model is not fit to choose d7/d14 either, and this
script says so rather than emitting a number.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                   "pi_bracket_spd_s0", "pi_scale_bracket.json")

# --- server reads on the withheld 2022 set (phase 29696) --------------------------------
# (scale actually shipped, winkler, coverage). d1 knob = SPD_SCALE_D1, d7/d14 = RECALIB_SPD.
OBS_2022 = {
    1:  [(0.90, 9.0314, 0.871), (1.00, 8.9825, 0.907)],   # 883905 / 883907
    # d7/d14: 1.10 = 883907, then the coverage-targeted move 884977. The 1.10->new
    # step was PRE-REGISTERED from the 1.10 anchor alone and predicted 17.62 / 15.16
    # against measured 17.601 / 15.409 -> +0.11% / -1.6%. See the control block below.
    7:  [(1.10, 17.809, 0.928), (1.03, 17.601, 0.900)],
    14: [(1.10, 16.452, 0.969), (0.90, 15.409, 0.890)],
}


def cv_curves(path=RES):
    """mean-over-folds CV curve per horizon: scales, coverage, winkler."""
    d = json.load(open(path))
    scales = np.array(d["scales"], float)
    out = {}
    for h in (1, 7, 14):
        cov, wk = [], []
        for s in d["scales"]:
            k = f"{s:.2f}"
            c = [f["by_h"][str(h)][k]["coverage"] for f in d["folds"].values()]
            w = [f["by_h"][str(h)][k]["winkler"] for f in d["folds"].values()]
            cov.append(np.mean(c)); wk.append(np.mean(w))
        out[h] = (scales, np.array(cov), np.array(wk))
    return out, list(d["folds"])


def s_at_cov(scales, cov, target):
    """CV scale whose CV coverage == target (monotone increasing in scale)."""
    if target <= cov[0] or target >= cov[-1]:
        return float("nan")
    return float(np.interp(target, cov, scales))


def predict(scales, cov, wk, s_ship, cov_obs, wk_obs):
    """Anchor the CV curve on one server point; return predicted optimum + curve."""
    s_eq = s_at_cov(scales, cov, cov_obs)
    if not np.isfinite(s_eq):
        return None
    rho = s_eq / s_ship                       # >1 : year errors WIDER than CV
    # level: the server's winkler at the anchor vs CV's winkler there
    wk_eq = float(np.interp(s_eq, scales, wk))
    lvl = wk_obs / wk_eq
    j = int(np.argmin(wk))
    s_cv_star = float(scales[j])
    # refine the CV argmin by a parabola through its neighbours
    if 0 < j < len(scales) - 1:
        y0, y1, y2 = wk[j - 1], wk[j], wk[j + 1]
        x0, x1, x2 = scales[j - 1], scales[j], scales[j + 1]
        den = (y0 - 2 * y1 + y2)
        if den > 0:
            s_cv_star = float(x1 - 0.5 * (x2 - x0) * (y2 - y0) / (2 * den))
    return dict(s_eq=s_eq, rho=rho, lvl=lvl, s_cv_star=s_cv_star,
                s_star=s_cv_star / rho,
                cov_star=float(np.interp(s_cv_star, scales, cov)),
                wk_star=float(np.interp(s_cv_star, scales, wk)) * lvl,
                wk_now=wk_obs,
                gain=wk_obs - float(np.interp(s_cv_star, scales, wk)) * lvl)


def real_to_cv(s_real, rho):
    return s_real * rho


def main():
    curves, folds = cv_curves()
    print(f"CV bracket: {len(folds)} folds {folds}\n")

    # ---------------- POSITIVE CONTROL at d1 -------------------------------------------
    print("=" * 78)
    print("POSITIVE CONTROL  (fit on the 2022 d1 scale-0.90 read, predict the 1.00 read)")
    print("=" * 78)
    scales, cov, wk = curves[1]
    (s_a, wk_a, cov_a), (s_b, wk_b, cov_b) = OBS_2022[1]
    p = predict(scales, cov, wk, s_a, cov_a, wk_a)
    if p is None:
        print("  anchor coverage outside the CV bracket -- control INCONCLUSIVE"); return
    # predict the OTHER point: real scale s_b -> CV-equivalent -> read CV curve, rescale
    s_b_cv = real_to_cv(s_b, p["rho"])
    pred_wk_b = float(np.interp(s_b_cv, scales, wk)) * p["lvl"]
    pred_cov_b = float(np.interp(s_b_cv, scales, cov))
    print(f"  anchor      : scale {s_a:.2f}  winkler {wk_a:.4f}  cov {cov_a*100:.1f}%")
    print(f"  fitted rho  : {p['rho']:.4f}   (CV-equivalent scale of the anchor {p['s_eq']:.4f})")
    print(f"  PREDICT @{s_b:.2f}: winkler {pred_wk_b:.4f}   cov {pred_cov_b*100:.1f}%")
    print(f"  MEASURED@{s_b:.2f}: winkler {wk_b:.4f}   cov {cov_b*100:.1f}%")
    e_w = pred_wk_b - wk_b
    e_c = (pred_cov_b - cov_b) * 100
    # the decision the model has to get right is the SIGN of the move
    sign_ok = (pred_wk_b < wk_a) == (wk_b < wk_a)
    print(f"  error       : winkler {e_w:+.4f} ({100*e_w/wk_b:+.2f}%)   coverage {e_c:+.2f} pp")
    print(f"  direction   : model says {'1.00 better' if pred_wk_b < wk_a else '0.90 better'};"
          f" truth says {'1.00 better' if wk_b < wk_a else '0.90 better'}  -> "
          f"{'PASS' if sign_ok else 'FAIL'}")
    ok = sign_ok and abs(e_w) / wk_b < 0.02
    print(f"  VERDICT     : {'PASS' if ok else 'FAIL'} "
          f"(need correct sign and <2% winkler error)\n")

    # ---------------- the actual question: d7 / d14 ------------------------------------
    print("=" * 78)
    print("PREDICTED 2022 OPTIMA")
    print("=" * 78)
    rows = []
    for h in (1, 7, 14):
        scales, cov, wk = curves[h]
        s_ship, wk_obs, cov_obs = OBS_2022[h][-1]
        p = predict(scales, cov, wk, s_ship, cov_obs, wk_obs)
        j = int(np.argmin(wk))
        print(f"\n-- horizon d{h}")
        print(f"   CV optimum        : scale {p['s_cv_star']:.3f}  cov {p['cov_star']*100:.1f}%"
              f"  (grid argmin {scales[j]:.2f}, winkler {wk[j]:.4f})")
        print(f"   shipped on 2022   : scale {s_ship:.2f}  winkler {wk_obs:.4f}  cov {cov_obs*100:.1f}%")
        # rho = CV scale needed to reach the year's observed coverage, over the shipped
        # scale. rho>1 => CV needs a WIDER interval for the same coverage => the year's
        # errors are NARROWER than CV's => narrow the shipped scale.
        print(f"   year inflation rho: {p['rho']:.4f}  "
              f"({'2022 errors NARROWER than CV' if p['rho']>1 else '2022 errors WIDER than CV'})")
        print(f"   => OPTIMAL SCALE  : {p['s_star']:.3f}   predicted winkler {p['wk_star']:.4f}"
              f"   GAIN {p['gain']:+.4f}")
        # predicted curve, and the safe side: Winkler is ASYMMETRIC about the optimum
        # (too wide costs width everywhere, too narrow costs 20x in the tail), so when the
        # step is a big extrapolation prefer a scale slightly WIDE of the argmin.
        print(f"   predicted curve (real scale -> winkler, coverage):")
        for st in (0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 1.00, 1.05, 1.10):
            scv = real_to_cv(st, p["rho"])
            if scv < scales[0] or scv > scales[-1]:
                continue
            w = float(np.interp(scv, scales, wk)) * p["lvl"]
            c = float(np.interp(scv, scales, cov))
            mark = "  <== argmin" if abs(st - p["s_star"]) < 0.013 else ""
            print(f"      {st:.3f} -> {w:8.4f}  cov {c*100:5.1f}%   ({w-wk_obs:+.4f}){mark}")
        rows.append((h, s_ship, p["s_star"], p["gain"], cov_obs, p["cov_star"]))

    print("\n" + "=" * 78)
    print("SUMMARY -- proposed RECALIB_SPD / SPD_SCALE_D1 for the next submission")
    print("=" * 78)
    print(f"{'h':>4} {'now':>7} {'proposed':>9} {'cov now':>9} {'cov target':>11} {'pred gain':>10}")
    tot = 0.0
    for h, s0, s1, g, c0, c1 in rows:
        print(f"{h:>4} {s0:>7.2f} {s1:>9.3f} {c0*100:>8.1f}% {c1*100:>10.1f}% {g:>+10.4f}")
        tot += g
    print(f"\n  total predicted speed-Winkler gain across horizons: {tot:+.4f}")
    print(f"  primary_score sensitivity ~0.0167 per unit speed_d1-equivalent; d7/d14 refs")
    print(f"  differ, so read the per-column gains above as the decision, not the sum.")


if __name__ == "__main__":
    main()
