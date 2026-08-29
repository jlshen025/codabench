#!/usr/bin/env python
"""Exact TreeSHAP feature attributions for the shipped HRES-MOS models.

Rubric dim 2 (Responsible & Transparent AI) asks for "interpretation of the results
and of what drives the model (e.g. feature importances via SHAP)". LightGBM computes
EXACT tree SHAP values natively (`pred_contrib=True`), so no extra dependency and no
sampling approximation.

Covers every shipped MOS model:
  * mean-MOS (lead, u|v) -> the direction driver at d+1
  * quantile-MOS (lead, tau) -> the speed q05/q50/q95 at d+1
Features: fcst_u, fcst_v, fcst_speed, lat, lon, woy_sin, woy_cos.

Also emits a SHAP *dependence* profile (mean SHAP of the top feature vs its own
decile) so the report can state the direction/strength of each effect, not just a
magnitude ranking.

Output: <EXP_OUTPUT_DIR>/shap_mos.json + a ready-to-paste markdown table.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KIT = f"{ROOT}/kit_phase2/phase_2"
for sub in ("", "/part0_dataset_setup", "/part1_forecast"):
    sys.path.insert(0, KIT + sub)
sys.path.insert(0, f"{ROOT}/scripts")

import joblib
import numpy as np
import pandas as pd
import forecast_hres as fh

OUT = os.environ.get("EXP_OUTPUT_DIR", ".")
BUNDLE = f"{ROOT}/scripts/results/mos_retrain_1619_c20_s0/mos_models.joblib"
FEATURES = fh.FEATURES
# Human-readable names + the physical quantity each stands for.
PHYS = {
    "fcst_speed": "HRES 10 m forecast wind speed (the driver's own magnitude)",
    "fcst_u":     "HRES 10 m zonal wind (west-east flow component)",
    "fcst_v":     "HRES 10 m meridional wind (south-north flow component)",
    "lat":        "latitude (fetch / distance-to-coast proxy along the N-S axis)",
    "lon":        "longitude (fetch / distance-to-coast proxy along the W-E axis)",
    "woy_sin":    "week-of-year (annual cycle, sine phase)",
    "woy_cos":    "week-of-year (annual cycle, cosine phase)",
}


def issue_dates(year, step=8):
    d0 = pd.Timestamp(f"{year}-01-01")
    out, d = [], d0
    while d + pd.Timedelta(days=14) <= pd.Timestamp(f"{year}-12-31"):
        out.append(d)
        d += pd.Timedelta(days=step)
    return out


def shap_table(model, X):
    """Exact TreeSHAP -> (mean|shap| per feature, signed mean shap per feature)."""
    contrib = model.booster_.predict(X.to_numpy(), pred_contrib=True)  # (n, F+1)
    sv = contrib[:, :-1]
    return (np.abs(sv).mean(axis=0), sv.mean(axis=0), sv)


def main() -> int:
    t0 = time.time()
    b = joblib.load(BUNDLE)
    qmos, mmos = b["qmos"], b["mmos"]
    print(f"bundle: train_years={b['train_years']} calib={b['calib_year']} "
          f"qmos={len(qmos)} mmos={len(mmos)}", flush=True)

    # Attribution sample: an independent year (2020 issues), subsampled for speed.
    iss = issue_dates(2020, step=8)
    df = fh.build_hres_table(iss)
    print(f"attribution frame: {len(df)} rows ({time.time()-t0:.0f}s)", flush=True)

    res, dep = {}, {}
    for key, model in list(mmos.items()) + list(qmos.items()):
        L, tgt = key
        sub = df[df["lead"] == L]
        if sub.empty:
            continue
        X = sub[FEATURES]
        absm, signed, sv = shap_table(model, X)
        name = f"lead{L}_{tgt}"
        res[name] = {f: dict(abs_shap=float(absm[i]), mean_shap=float(signed[i]),
                             share=float(absm[i] / absm.sum()))
                     for i, f in enumerate(FEATURES)}
        # dependence profile of the dominant feature (deciles of that feature)
        top = FEATURES[int(np.argmax(absm))]
        q = pd.qcut(X[top], 10, labels=False, duplicates="drop")
        dep[name] = dict(feature=top,
                         bin_center=[float(v) for v in X[top].groupby(q).mean()],
                         mean_shap=[float(v) for v in pd.Series(sv[:, FEATURES.index(top)])
                                    .groupby(q.to_numpy()).mean()])
        print(f"  {name:14s} top={top:11s} "
              + " ".join(f"{f}={absm[i]:.3f}" for i, f in enumerate(FEATURES)), flush=True)

    # ---- markdown, ready to paste into the report -------------------------------
    lines = ["| model | " + " | ".join(FEATURES) + " |",
             "|---" * (len(FEATURES) + 1) + "|"]
    for name, d in res.items():
        lines.append(f"| {name} | " +
                     " | ".join(f"{d[f]['share']*100:.1f}%" for f in FEATURES) + " |")
    md = "\n".join(lines)

    payload = dict(bundle=BUNDLE, train_years=b["train_years"], calib_year=b["calib_year"],
                   n_rows=int(len(df)), features=FEATURES, physical_meaning=PHYS,
                   shap=res, dependence=dep, markdown=md)
    with open(os.path.join(OUT, "shap_mos.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print("\n" + md)
    print(f"\n[{time.time()-t0:.0f}s] wrote {OUT}/shap_mos.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
