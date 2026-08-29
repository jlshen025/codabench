#!/usr/bin/env python
"""Regenerate the Phase-2 forecast deliverable (`predictions.csv`) from an inference
directory ALONE — the final-evaluation window's operational requirement.

WHY THIS EXISTS. The shipped v11b was produced as a chain of zip patches: each stage
read a hardcoded prior submission zip and looped `range(1, 9)`. That chain can CLIMB
but cannot RE-RUN — on a fresh inference directory there is no base zip to patch. This
module is the end-to-end path: inference dir + frozen fitted artifacts -> validated zip,
with no reference to any previous submission.

    STAGE            WHAT IT DOES                                   WHERE IT RUNS
    era5    ERA5 init per window from the public WeatherBench2      1 CPU task, ~2 min
            GCS zarr at each window's context_end (Gate-1 fair:     (needs internet; O1
            ERA5 only, nothing past context_end)                     proved compute nodes
                                                                     have it here)
    ens     M-1 coherent smooth wind IC perturbations per window     1 CPU task, ~2 min
    pangu   Pangu-Weather ONNX rollout of every (window, member)     SLURM ARRAY,
            -> coarse-grid driver features                           ~36.5 GB/task
    assemble  base CSV -> d7/d14 speed-PI widening -> ens-mean      1 CPU task, ~25 min
            direction -> d1 arc map -> ctx d1 speed -> schema        ~12 GB
            check -> zip

The four fitted artifacts are FROZEN (trained on 2016-2020, never refit here) and live
on backed-up /project in `models_final/`: mos_models.joblib, ens_rebuild_fit.json,
ens_spdcond_fit.json, mos_ctx_bundle.joblib, plus the frozen climatology + footprint
order. Scratch is purgeable and must never be the only copy.

GUARDS THAT MUST SURVIVE ANY REBUILD (each one cost a measured regression to learn):
  * SHIP_LEADS=1 — the d7 arc conditioner is LOO-positive and withheld-year NEGATIVE
    (+15 / +32 on the server). The allow-list, not the fit flag, decides what ships.
  * The four patch stages align POSITIONALLY inside a horizon mask, so the row order
    (window-major -> horizon -> hour -> footprint) is load-bearing. p2_predcsv.validate
    re-checks it after every stage.
  * Row counts are derived from the resolved window set, never a literal 4,196,640.
  * The d7/d14 speed PI carries a x1.10 widening (p2_recalib) calibrated on a server coverage
    read. It is invisible in the code that computes those quantiles and was missing from the
    first end-to-end draft; the exact-reproduction test is what caught it.

USAGE
  python p2_rebuild_predictions.py --stage plan       --inference-dir DIR
  python p2_rebuild_predictions.py --stage era5       --inference-dir DIR --work W
  python p2_rebuild_predictions.py --stage ens        --work W --members 8
  python p2_rebuild_predictions.py --stage pangu      --work W --shard 0/32
  python p2_rebuild_predictions.py --stage assemble   --inference-dir DIR --work W --out OUT.zip
  python p2_rebuild_predictions.py --stage all        ...   # serial; dry-runs only
  python p2_rebuild_predictions.py --stage verify     --out OUT.zip --against REF.zip
  python p2_rebuild_predictions.py --stage assemble --no-fm ...   # FM-free fallback, ~4 min
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.environ.get("SEAWINDS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = f"{ROOT}/scripts"
MODELS_DEFAULT = f"{ROOT}/models_final"
sys.path.insert(0, SCRIPTS)


# --------------------------------------------------------------------------- env
def resolve_env(a) -> dict:
    """Bind every path the stage modules read, BEFORE they are imported (they capture
    their configuration at import time). Returns the resolved map for the run log."""
    work = os.path.abspath(a.work)
    models = os.path.abspath(a.models)
    fm = f"{work}/fm"
    env = {
        "SEAWINDS_INFER_DIR": os.path.abspath(a.inference_dir) if a.inference_dir else "",
        "SEAWINDS_FM_DIR": fm,
        "FM_ERA5_INIT_DIR": f"{fm}/era5_init",
        "FM_ENS_INIT_DIR": f"{fm}/ens_init",
        "FM_ENS_EVAL_DIR": a.ens_feats or f"{fm}/ens_feats",
        "FM_WEIGHTS_DIR": a.weights,
        # frozen fitted artifacts (durable /project copies)
        "P2_MOS_MODELS": f"{models}/mos_models.joblib",
        "ENS_REBUILD_FIT": f"{models}/ens_rebuild_fit.json",
        "ENS_SPDCOND_FIT": f"{models}/ens_spdcond_fit.json",
        "MOS_CTX_BUNDLE": f"{models}/mos_ctx_bundle.joblib",
        "SEAWINDS_CLIM_NPZ": f"{models}/clim_2016_2020.npz",
        "SEAWINDS_FOOTPRINT_ORDER": f"{models}/footprint_order.parquet",
        # intermediate + final artifacts
        "P2_BASE_OUTDIR": f"{work}/s1_base",
        "ENS_REBUILD_SRC_ZIP": f"{work}/s1b_recal/predictions.csv",
        "ENS_REBUILD_OUT_CSV": f"{work}/s2_ensdir/predictions.csv",
        "SPDCOND_SRC_ZIP": f"{work}/s2_ensdir/predictions.csv",
        "SPDCOND_OUT_CSV": f"{work}/s3_d1arc/predictions.csv",
        "SRC_ZIP": f"{work}/s3_d1arc/predictions.csv",
        "CTX_OUT_CSV": f"{work}/s4_ctx/predictions.csv",
        "CTX_OUT_ZIP": os.path.abspath(a.out) if a.out else "",
        # THE GUARD: never widen this without a withheld-year server read .
        "SHIP_LEADS": a.ship_leads,
        "BUILD_TAG": a.tag,
        "EXP_OUTPUT_DIR": os.environ.get("EXP_OUTPUT_DIR", work),
    }
    env = {k: v for k, v in env.items() if v}
    os.environ.update(env)
    os.environ.setdefault("PHASE2_DATA_ROOT", os.environ["SEAWINDS_PHASE2_DIR"])
    os.makedirs(work, exist_ok=True)
    return env


# What the build actually CONSUMES from an inference window. Extra columns are fine;
# these are the ones whose absence or renaming would break the wave, and the 08-08 set is
# authored by the organizers, not by us.
REQ_REANALYSIS = ["latitude", "longitude", "u10", "v10", "u100", "v100"]
REQ_HRES = (["latitude", "longitude"]
            + [f"fcst_{v}_d{L}_h{h}" for L in (1, 7) for h in (0, 6, 12, 18)
               for v in ("speed", "dir")])
REQ_META = ["id", "context_start", "context_end"]
NGRID = 2565            # the 0.25 deg coarse grid the MOS is fitted on
CTX_DAYS, CTX_HOURS = 14, 4


def check_inference_schema(inf: str) -> list[str]:
    """Compare a fresh inference directory against the schema the pipeline consumes.

    The final-evaluation set is authored by the organizers and released days before the
    deadline. A silently renamed column or a different grid would otherwise surface as an
    obscure failure deep inside a 48-minute wave; this turns that into an up-front,
    itemised go/no-go. Returns a list of problems (empty = good)."""
    import pandas as pd
    import p2_windows as W
    problems = []
    for wid in W.window_ids(inf):
        w = f"{inf}/window_{wid}"
        try:
            md = json.load(open(f"{w}/metadata.json"))
            for k in REQ_META:
                if k not in md:
                    problems.append(f"window_{wid}/metadata.json: missing key '{k}'")
        except Exception as e:
            problems.append(f"window_{wid}/metadata.json: unreadable ({e})")
        for fname, req, nrow in (
                ("context_reanalysis_north_sea.parquet", REQ_REANALYSIS,
                 NGRID * CTX_HOURS * CTX_DAYS),
                ("context_hres_north_sea.parquet", REQ_HRES, NGRID)):
            try:
                d = pd.read_parquet(f"{w}/{fname}")
            except Exception as e:
                problems.append(f"window_{wid}/{fname}: unreadable ({e})"); continue
            miss = [c for c in req if c not in d.columns]
            if miss:
                problems.append(f"window_{wid}/{fname}: missing columns {miss}")
            npts = d[["latitude", "longitude"]].drop_duplicates().shape[0] \
                if {"latitude", "longitude"} <= set(d.columns) else -1
            if npts != NGRID:
                problems.append(f"window_{wid}/{fname}: {npts} unique grid points, expected {NGRID}")
            if len(d) != nrow:
                problems.append(f"window_{wid}/{fname}: {len(d)} rows, expected {nrow}")
    return problems


def preflight(env, need_models=True):
    """Fail loudly and early on a missing input, rather than 40 minutes into a stage."""
    missing = []
    inf = env.get("SEAWINDS_INFER_DIR")
    if inf:
        import p2_windows as W
        ids = W.window_ids(inf)
        for wid in ids:
            for f in W.REQUIRED_FILES:
                p = f"{inf}/window_{wid}/{f}"
                if not os.path.exists(p):
                    missing.append(p)
        print(f"inference dir {inf}: {len(ids)} windows {ids}", flush=True)
        missing += check_inference_schema(inf)
    if need_models:
        for k in ("P2_MOS_MODELS", "ENS_REBUILD_FIT", "ENS_SPDCOND_FIT",
                  "MOS_CTX_BUNDLE", "SEAWINDS_FOOTPRINT_ORDER"):
            if not os.path.exists(env[k]):
                missing.append(f"{k}={env[k]}")
        # the climatology is auto-buildable from the per-year AROME cache, so it is a
        # warning here -- but only if that (purgeable) cache is actually present.
        if not os.path.exists(env["SEAWINDS_CLIM_NPZ"]):
            cache = os.environ.get("SEAWINDS_AROME_CACHE",
                                   "./work/cache/arome_footprint")
            years = [y for y in (2016, 2017, 2018, 2019, 2020)
                     if not os.path.exists(f"{cache}/arome_fp_{y}.npz")]
            if years:
                missing.append(f"SEAWINDS_CLIM_NPZ={env['SEAWINDS_CLIM_NPZ']} (and cannot "
                               f"rebuild it: {cache} is missing years {years})")
            else:
                print(f"WARN frozen climatology absent; will build from {cache} "
                      f"and freeze it", flush=True)
    assert not missing, "PREFLIGHT FAILED, missing:\n  " + "\n  ".join(missing)
    print("preflight OK", flush=True)


def _windows():
    import p2_windows as W
    return W.windows()


# ------------------------------------------------------------------------- stages
def stage_freeze(models: str):
    """One-off: put the two inference-independent artifacts on backed-up storage, so no
    rebuild ever depends on the purgeable scratch cache."""
    import shutil
    import p2_forecast_cv as FC
    import p2_siting as S
    os.makedirs(models, exist_ok=True)
    dst = f"{models}/footprint_order.parquet"
    if not os.path.exists(dst):
        src = f"{S.CACHE}/footprint_order.parquet"
        shutil.copy2(src, dst)
        print(f"froze footprint order {src} -> {dst}", flush=True)
    npz = os.environ["SEAWINDS_CLIM_NPZ"]
    if not os.path.exists(npz):
        t0 = time.time()
        FC.clim_to_npz(FC.build_clim([2016, 2017, 2018, 2019, 2020]), npz)
        print(f"froze climatology -> {npz} "
              f"({os.path.getsize(npz)/1e6:.0f}MB, {time.time()-t0:.0f}s)", flush=True)
    # prove the frozen copy round-trips to the same arrays the builder produces
    c = FC.clim_from_npz(npz)
    print(f"climatology OK: {len(c)} (month,hour) cells, "
          f"{len(c[(1, 0)]['q05'])} footprint points", flush=True)


def stage_era5():
    import fm_fetch_era5
    fm_fetch_era5.main()


def stage_ens(members: int):
    """Perturbed IC members. Seeded by WINDOW ID order (numeric), so the member set is
    reproducible regardless of how many windows the eval set ships (a lexical sort puts
    window_10 before window_2 and would silently re-seed everything)."""
    import fm_ens_build as EB
    import p2_windows as W
    fm = os.environ["SEAWINDS_FM_DIR"]
    init_dir = os.environ["FM_ERA5_INIT_DIR"]
    ens_dir = os.environ["FM_ENS_INIT_DIR"]
    os.makedirs(ens_dir, exist_ok=True)
    import numpy as np
    manifest = []
    for ii, wid in enumerate(W.window_ids()):
        src = f"{init_dir}/window_{wid}_init.npz"
        assert os.path.exists(src), f"missing ERA5 init {src} — run --stage era5 first"
        for m in range(members):
            tag = f"window_{wid}_m{m}"
            dst = f"{ens_dir}/{tag}_init.npz"
            if not os.path.exists(dst):
                if m == 0:
                    np.savez(dst, **dict(np.load(src)))        # member 0 = control
                else:
                    EB.perturb(src, dst, seed=1000 * m + ii)
            manifest.append({"tag": tag, "init_npz": dst})
    mpath = f"{fm}/ens_manifest_rebuild.json"
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"{len(manifest)} rollouts ({members} members x {len(manifest)//members} windows)"
          f" -> {mpath}", flush=True)
    return mpath


def stage_pangu(shard: str):
    """One SLURM array task: roll its slice of the ensemble manifest."""
    fm = os.environ["SEAWINDS_FM_DIR"]
    mpath = f"{fm}/ens_manifest_rebuild.json"
    assert os.path.exists(mpath), f"missing {mpath} — run --stage ens first"
    i, n = (int(x) for x in shard.split("/"))
    out = f"{os.environ['FM_ENS_EVAL_DIR']}/ens_feats_{i:03d}.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sys.argv = ["fm_pangu_infer", "--manifest", mpath, "--out", out, "--shard", shard,
                "--device", os.environ.get("FM_DEVICE", "cpu")]
    import fm_pangu_infer
    fm_pangu_infer.main()


def stage_assemble(skip_base=False, no_fm=False):
    """base CSV -> d7/d14 speed-PI widening -> ens-mean d7/d14 direction -> d1 arc map
    -> ctx d1 speed -> zip.

    no_fm=True skips stages 3 and 4, the ONLY two that consume the Pangu ensemble, and
    produces a complete schema-legal submission from the MOS + climatology + context
    features alone. That is the fallback for the single point of failure in the 08-08
    wave: every direction column we ship flows through one ERA5-fetch -> Pangu-rollout
    path, and that path meets unseen dates on the day. It costs ~4 minutes and no array
    job, so a broken FM path degrades the entry to roughly the v3-era direction skill
    instead of costing the dimension."""
    import p2_predcsv as PC
    t0 = time.time()
    wins = _windows()
    nw = len(wins)
    steps = []

    os.makedirs(f"{os.path.dirname(os.environ['ENS_REBUILD_SRC_ZIP'])}", exist_ok=True)
    if not skip_base or not os.path.exists(os.environ["P2_BASE_OUTDIR"] + "/predictions.csv"):
        print(f"\n=== [1/5] base CSV (MOS d1 spd+dir, MOS d7 dir, climatology elsewhere)",
              flush=True)
        import p2_build_predictions as BP
        BP.main()
        steps.append("base")
    else:
        print("\n=== [1/5] base CSV — reusing existing", flush=True)
    v = PC.read_predictions(os.environ["P2_BASE_OUTDIR"] + "/predictions.csv")
    print("   base:", PC.validate(v, nw), flush=True)
    del v

    print(f"\n=== [2/5] speed-PI recalibration (server-coverage scales)  "
          f"({time.time()-t0:.0f}s)", flush=True)
    import p2_recalib
    _spd = os.environ.get("RECALIB_SPD", "1.0,1.1,1.1")
    # GUARD: RECALIB_SPD's FIRST element (lead 1) is INERT — stage [5/5]
    # p2_mos_ctx_ship.apply_ rewrites every lead-1 q05/q50/q95 row, in the --no-fm
    # branch too, so a d1 scale set here is silently discarded. The live d1 knob is
    # SPD_SCALE_D1 (p2_mos_ctx_ship.py). Fail loudly rather than no-op: a silently
    # ignored operating point under endgame time pressure is the failure shape.
    if abs(float(_spd.split(",")[0]) - 1.0) > 1e-9:
        raise SystemExit(
            f"RECALIB_SPD={_spd}: the d1 element ({_spd.split(',')[0]}) is INERT — stage "
            f"[5/5] p2_mos_ctx_ship overwrites all lead-1 speed rows. Set SPD_SCALE_D1 "
            f"instead, and leave RECALIB_SPD's first element at 1.0.")
    sys.argv = ["p2_recalib", "--in", os.environ["P2_BASE_OUTDIR"] + "/predictions.csv",
                "--out", os.environ["ENS_REBUILD_SRC_ZIP"],
                "--spd", _spd,
                "--dir", os.environ.get("RECALIB_DIR", "1.0,1.0,1.0")]
    p2_recalib.main()
    PC.validate(PC.read_predictions(os.environ["ENS_REBUILD_SRC_ZIP"]), nw)

    if no_fm:
        print("\n=== [3/5, 4/5] SKIPPED (--no-fm): both Pangu-ensemble direction stages.\n"
              "    Direction stays MOS(d1,d7) + climatology(d14); speed is unaffected.",
              flush=True)
        os.environ["SRC_ZIP"] = os.environ["ENS_REBUILD_SRC_ZIP"]
    else:
        print(f"\n=== [3/5] ensemble-MEAN d7+d14 direction centre  ({time.time()-t0:.0f}s)",
              flush=True)
        import fm_ens_rebuild
        fm_ens_rebuild.apply_()

        print(f"\n=== [4/5] speed-conditioned d1 direction arc  ({time.time()-t0:.0f}s)",
              flush=True)
        import fm_dir_spdcond
        fm_dir_spdcond.apply_()

    print(f"\n=== [5/5] context-feature d1 SPEED MOS  ({time.time()-t0:.0f}s)", flush=True)
    import p2_mos_ctx_ship
    p2_mos_ctx_ship.apply_()

    final = os.environ.get("CTX_OUT_ZIP") or os.environ["CTX_OUT_CSV"]
    v = PC.read_predictions(final)
    summary = PC.validate(v, nw)
    print(f"\nFINAL {final}\n  {summary}\n  ({time.time()-t0:.0f}s total)", flush=True)
    json.dump({"artifact": final, "windows": nw, "ship_leads": os.environ.get("SHIP_LEADS"),
               "no_fm": bool(no_fm), "summary": summary,
               "seconds": round(time.time() - t0)},
              open(f"{os.environ['EXP_OUTPUT_DIR']}/rebuild_manifest.json", "w"), indent=1)
    return final


def stage_fmverify(against: str):
    """Acceptance test for the FM half: the regenerated ensemble driver features must
    match the ones behind the shipped submission. The ens-MEAN direction centre is the
    campaign's largest single gain, so a silently-different ERA5 init or perturbation
    seed would degrade it with nothing downstream to notice."""
    import numpy as np
    import fm_ens_rebuild as ER
    import p2_windows as W
    new = ER.load_ens([os.environ["FM_ENS_EVAL_DIR"]])
    ref = ER.load_ens([against])
    tags = sorted(set(new["tag"]))
    ref = ref[ref["tag"].isin(tags)]
    print(f"comparing {len(tags)} rollout tags: {tags}", flush=True)
    key = ["tag", "lat", "lon", "lead", "hour"]
    cols = ["u10", "v10", "u1000", "v1000", "u925", "v925"]
    a = new.sort_values(key).reset_index(drop=True)
    b = ref.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b), f"row counts differ: regenerated {len(a)} vs reference {len(b)}"
    for k in key:
        assert (a[k].to_numpy() == b[k].to_numpy()).all(), f"{k} misaligned"
    rep, worst = {}, 0.0
    for c in cols:
        d = np.abs(a[c].to_numpy() - b[c].to_numpy())
        rep[c] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}
        worst = max(worst, float(d.max()))
    # a bit-identical ONNX rollout is the expectation; 1e-4 m/s absorbs only library-level
    # float reassociation, far below any change the ensemble mean could feel.
    ok = worst < 1e-4
    print(json.dumps(rep, indent=1))
    print(f"\nworst |delta| = {worst:.3e} m/s -> "
          f"{'FM PATH REPRODUCES the shipped features' if ok else 'FM PATH DIFFERS'}")
    json.dump({"against": against, "tags": tags, "worst_abs": worst, "identical": bool(ok),
               "report": rep},
              open(f"{os.environ.get('EXP_OUTPUT_DIR', '.')}/fmverify_report.json", "w"), indent=1)
    return ok


def stage_sentinel(out: str, delta: float = 1.0):
    """Known-answer NEGATIVE control for the acceptance test's comparator.

    `stage_verify` reporting "identical" is only evidence if the comparator can detect a
    difference at all — a compare that always returns 0 diffs would pass the acceptance test
    vacuously. So: inject a known perturbation of exactly `delta` into one column of the
    accepted artifact and require the comparator to report exactly that, on exactly the
    expected number of rows, and nothing on any other column."""
    import numpy as np
    import p2_predcsv as PC
    a = PC.read_predictions(out)
    b = a.copy()
    m = (b["horizon"] == 7).to_numpy()
    b.loc[m, "q50"] = b.loc[m, "q50"].to_numpy() + delta
    rep = PC.compare(a, b)
    got = rep["q50"]["max_abs"]
    n_expect = int(m.sum())
    others = {k: v for k, v in rep.items()
              if not k.startswith("_") and k != "q50" and v["n_diff"] != 0}
    ok = (abs(got - delta) < 1e-9 and rep["q50"]["n_diff"] == n_expect and not others
          and rep["_identical"] is False)
    print(f"injected delta={delta} on horizon 7 q50 ({n_expect:,} rows)")
    print(f"comparator reported max_abs={got} n_diff={rep['q50']['n_diff']:,} "
          f"identical={rep['_identical']} other-columns-touched={list(others)}")
    print("SENTINEL PASS — the comparator detects a known difference and localizes it"
          if ok else "SENTINEL FAIL — the acceptance test's 'identical' verdict is not evidence")
    json.dump({"injected_delta": delta, "max_abs": got, "n_diff": rep["q50"]["n_diff"],
               "n_expected": n_expect, "other_columns_touched": list(others),
               "identical_flag": rep["_identical"], "pass": bool(ok)},
              open(f"{os.environ.get('EXP_OUTPUT_DIR', '.')}/sentinel_report.json", "w"), indent=1)
    return ok


def stage_verify(out: str, against: str):
    """Acceptance test: the rebuilt artifact must equal a reference artifact exactly."""
    import p2_predcsv as PC
    a = PC.read_predictions(out)
    b = PC.read_predictions(against)
    rep = PC.compare(a, b)
    print(json.dumps(rep, indent=1))
    ok = rep["_identical"]
    print(f"\n{'IDENTICAL — acceptance test PASSED' if ok else 'DIFFERS — see report above'}")
    json.dump({"out": out, "against": against, "identical": bool(ok), "report": rep},
              open(f"{os.environ.get('EXP_OUTPUT_DIR', '.')}/verify_report.json", "w"), indent=1)
    return ok


def stage_plan(a, env, members: int, shards: int):
    import p2_windows as W
    wins = W.windows()
    n = len(wins) * members
    print(f"""
REBUILD PLAN
  inference dir : {env.get('SEAWINDS_INFER_DIR')}
  windows       : {len(wins)}  -> submission `window` column 0..{len(wins)-1}
  rollouts      : {len(wins)} windows x {members} members = {n}
  array shards  : {shards}  ({-(-n // shards)} rollouts/task)
  rows expected : {43715 * len(wins) * 3 * 4:,}
  work dir      : {a.work}
  final zip     : {a.out}
  SHIP_LEADS    : {env.get('SHIP_LEADS')}   (1 = d1 only; NEVER add 7 — A11/A13)

SLURM WAVE — submit as dependency-chained sbatch jobs. Limits are the
DRESS-REHEARSAL MEASUREMENTS (dress_*_s0, 8 windows x 8 members), not estimates:
  A  final_prep   1 task  cpu 4  24G  0:20:00   --stage era5 then --stage ens
                  measured 2:54 (8 ERA5 inits + 64 ICs); writes ~20 GB of npz
  B  final_roll   array 0-{shards-1}%12  cpu 8  45G  1:15:00
                  --stage pangu --shard $SLURM_ARRAY_TASK_ID/{shards}
                  measured 42:11 wall for the whole array; slowest shard 39 min;
                  peak RSS 36.4 GB -> 45 G is correct, do not trim
  C  final_asm    1 task  cpu 8  48G  0:40:00   --stage assemble   (afterok:B)
                  measured 3:08
  TOTAL measured wall-clock end-to-end: 48 min.
Then: --stage verify against the previous submission to see exactly what moved, and
p2_predcsv.validate has already run inside assemble. Preflight schema-checks the new
inference dir first and fails with an itemised list if the organizers changed anything.
""", flush=True)
    return wins


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["plan", "freeze", "era5", "ens", "pangu", "assemble",
                             "all", "verify", "fmverify", "sentinel"])
    ap.add_argument("--inference-dir", default="")
    ap.add_argument("--work", default="./work/rebuild")
    ap.add_argument("--models", default=MODELS_DEFAULT)
    ap.add_argument("--out", default="")
    ap.add_argument("--against", default="", help="verify: reference zip/csv to match")
    ap.add_argument("--ens-feats", default="", help="reuse an existing ens_feats_*.parquet dir")
    ap.add_argument("--weights", default=f"{os.environ.get('AUTOAGENT_CACHE_DIR', '')}/pangu")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--shard", default="", help="pangu: i/n")
    ap.add_argument("--tag", default="rebuild")
    ap.add_argument("--ship-leads", default="1")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--no-fm", action="store_true",
                    help="skip both Pangu-ensemble direction stages: the FM-free fallback")
    a = ap.parse_args()

    env = resolve_env(a)
    print("resolved env:\n  " + "\n  ".join(f"{k}={v}" for k, v in sorted(env.items())), flush=True)

    if a.stage == "verify":
        sys.exit(0 if stage_verify(a.out, a.against) else 1)
    if a.stage == "fmverify":
        sys.exit(0 if stage_fmverify(a.against) else 1)
    if a.stage == "sentinel":
        sys.exit(0 if stage_sentinel(a.out) else 1)
    if a.stage == "freeze":
        stage_freeze(os.path.abspath(a.models)); return
    if a.stage == "plan":
        preflight(env); stage_plan(a, env, a.members, a.shards); return
    if a.stage == "era5":
        preflight(env, need_models=False); stage_era5(); return
    if a.stage == "ens":
        preflight(env, need_models=False); stage_ens(a.members); return
    if a.stage == "pangu":
        assert a.shard, "--stage pangu needs --shard i/n"
        stage_pangu(a.shard); return
    if a.stage == "assemble":
        preflight(env); stage_assemble(a.skip_base, a.no_fm); return
    if a.stage == "all":
        preflight(env, need_models=False)
        stage_era5(); stage_ens(a.members)
        n = len(_windows()) * a.members
        for i in range(a.shards):
            if i < n:
                stage_pangu(f"{i}/{a.shards}")
        preflight(env); stage_assemble(a.skip_base, a.no_fm)


if __name__ == "__main__":
    main()
