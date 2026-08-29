#!/usr/bin/env python
"""Compute-footprint accounting for the Phase-2 report (rubric dim 2: Environmental cost).

CodeCarbon-style bottom-up estimate from SLURM accounting (`sacct`), over exactly the
jobs this project submitted (job ids read from each run-dir's provenance.json, so no
other project's work is counted).

Energy model (every constant is stated so a reader can re-scale it):
    E_cpu  = alloc_cores * W_PER_CORE * hours
    E_ram  = alloc_GB    * W_PER_GB   * hours          (CodeCarbon's 0.375 W/GB)
    E_site = (E_cpu + E_ram) * PUE
    CO2e   = E_site * grid_intensity

Hardware: Alliance Canada `nibi` (SHARCNET, Waterloo, Ontario) CPU nodes —
2 x Intel Xeon 6 "Granite Rapids", 96 cores/socket, 192 cores + 766 GB per node.
A 96-core Granite Rapids part is a 500 W TDP class chip -> 500/96 = 5.2 W/core.

Reports allocated (reserved) core-hours AND actually-consumed CPU-hours (sacct TotalCPU);
the honest environmental number is the ALLOCATED one (the cores are held either way).

Usage:  python scripts/p2_compute_footprint.py [--since 2026-07-01]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

W_PER_CORE = 5.2       # 500 W TDP / 96 cores, Intel Xeon 6 (Granite Rapids)
W_PER_GB = 0.375       # CodeCarbon DRAM model
PUE = 1.2              # modern air/water-cooled HPC hall (Alliance Canada)
# Ontario grid carbon intensity: ~92 % non-emitting (nuclear + hydro).
GRID_G_PER_KWH = {"Ontario (IESO)": 30.0, "Canada avg": 110.0,
                  "France (RTE)": 56.0, "EU-27 avg": 242.0, "Germany": 380.0}

# Job-name -> pipeline stage. Order matters (first match wins).
STAGE_RULES = [
    (r"^(kit_|fe_|heavy|cv_|cvdir|cvst|dirlvl|station|ens_cvab|dircond|patch)", "phase1_legacy"),
    (r"fpcache|cache", "data_prep"),
    (r"siting", "siting_sim"),
    (r"pangu|fm_ens_(build|rebuild)|fm_pangu|ens_(cv2020|train|eval)|_roll", "fm_inference"),
    (r"mos|clim", "forecast_train"),
    (r"dir_|dirasym|asym|spdcond|condcompare|refine|finalbuild|d14", "forecast_cv"),
    (r"build_|preds|predictions|v9|v10|_asm|_prep", "submission_build"),
]
# Which stages are "training/fitting" vs "inference/prediction" (rubric asks for the split)
TRAIN_STAGES = {"forecast_train", "forecast_cv", "siting_sim", "data_prep", "phase1_legacy"}


def stage_of(name: str) -> str:
    for pat, st in STAGE_RULES:
        if re.search(pat, name, re.I):
            return st
    return "other"


def parse_elapsed(s: str) -> float:
    """SLURM elapsed -> hours."""
    d = 0
    if "-" in s:
        d, s = s.split("-", 1)
        d = int(d)
    p = [float(x) for x in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0.0)
    return d * 24 + p[0] + p[1] / 60 + p[2] / 3600


def parse_mem(tres: str) -> float:
    """AllocTRES -> GB."""
    m = re.search(r"mem=([\d.]+)([MGT])", tres)
    if not m:
        return 0.0
    v, u = float(m.group(1)), m.group(2)
    return v / 1024 if u == "M" else v * 1024 if u == "T" else v


def parse_gpu(tres: str) -> int:
    m = re.search(r"gres/gpu=(\d+)", tres)
    return int(m.group(1)) if m else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="", help="ISO date; only jobs submitted on/after")
    ap.add_argument("--out", default="scripts/results/compute_footprint.json")
    a = ap.parse_args()

    ids = sorted({json.load(open(p)).get("job_id")
                  for p in glob.glob("scripts/results/*/provenance.json")}
                 - {None, ""})
    raw = subprocess.run(
        ["sacct", "-j", ",".join(ids), "--noheader", "--parsable2",
         "--format=JobID,JobName%40,State,Submit,Elapsed,AllocCPUS,AllocTRES%80,CPUTimeRAW,TotalCPU"],
        capture_output=True, text=True, check=True).stdout

    jobs = []
    for line in raw.splitlines():
        f = line.split("|")
        if len(f) < 9 or "." in f[0]:      # keep the parent step only
            continue
        jid, name, state, submit, elapsed, ncpu, tres, cputime_raw, totalcpu = f[:9]
        if a.since and submit < a.since:
            continue
        h = parse_elapsed(elapsed)
        ncpu = int(ncpu or 0)
        gb = parse_mem(tres)
        ngpu = parse_gpu(tres)
        e_cpu = ncpu * W_PER_CORE * h / 1000.0            # kWh
        e_ram = gb * W_PER_GB * h / 1000.0
        jobs.append(dict(job_id=jid, name=name, state=state, submit=submit,
                         stage=stage_of(name), hours=h, cores=ncpu, gb=gb, gpus=ngpu,
                         core_hours=ncpu * h, gpu_hours=ngpu * h,
                         cpu_hours_used=parse_elapsed(totalcpu or "0:0:0"),
                         kwh_it=e_cpu + e_ram))

    by_stage = defaultdict(lambda: dict(n=0, hours=0.0, core_hours=0.0, gpu_hours=0.0,
                                        cpu_hours_used=0.0, kwh_it=0.0))
    for j in jobs:
        s = by_stage[j["stage"]]
        s["n"] += 1
        for k in ("hours", "core_hours", "gpu_hours", "cpu_hours_used", "kwh_it"):
            s[k] += j[k]

    tot_kwh_it = sum(j["kwh_it"] for j in jobs)
    tot_kwh = tot_kwh_it * PUE
    train_kwh = sum(j["kwh_it"] for j in jobs if j["stage"] in TRAIN_STAGES) * PUE
    infer_kwh = tot_kwh - train_kwh

    out = dict(
        n_jobs=len(jobs), since=a.since or "all",
        constants=dict(w_per_core=W_PER_CORE, w_per_gb=W_PER_GB, pue=PUE,
                       node="2x Intel Xeon 6 Granite Rapids, 192 cores / 766 GB (nibi, SHARCNET)"),
        totals=dict(core_hours=sum(j["core_hours"] for j in jobs),
                    cpu_hours_used=sum(j["cpu_hours_used"] for j in jobs),
                    gpu_hours=sum(j["gpu_hours"] for j in jobs),
                    wall_hours=sum(j["hours"] for j in jobs),
                    kwh_it=tot_kwh_it, kwh_site=tot_kwh,
                    kwh_train=train_kwh, kwh_inference=infer_kwh),
        kgco2e={k: tot_kwh * v / 1000.0 for k, v in GRID_G_PER_KWH.items()},
        by_stage={k: v for k, v in sorted(by_stage.items(), key=lambda x: -x[1]["kwh_it"])},
        jobs=sorted(jobs, key=lambda j: -j["core_hours"]),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"jobs={out['n_jobs']}  since={out['since']}")
    print(f"allocated core-hours = {out['totals']['core_hours']:.1f}"
          f"   (CPU-hours actually consumed = {out['totals']['cpu_hours_used']:.1f})")
    print(f"GPU-hours = {out['totals']['gpu_hours']:.1f}")
    print(f"energy: IT {tot_kwh_it:.2f} kWh -> site {tot_kwh:.2f} kWh @PUE {PUE}"
          f"   [train {train_kwh:.2f} / inference {infer_kwh:.2f}]")
    for k, v in out["kgco2e"].items():
        print(f"   CO2e @ {k:16s} {GRID_G_PER_KWH[k]:5.0f} gCO2e/kWh -> {v:7.3f} kgCO2e")
    print("\nby stage:")
    print(f"  {'stage':18s} {'n':>3s} {'core-h':>9s} {'kWh_IT':>8s}")
    for k, v in out["by_stage"].items():
        print(f"  {k:18s} {v['n']:3d} {v['core_hours']:9.1f} {v['kwh_it']:8.2f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
