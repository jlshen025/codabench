# Sea Winds Predictions 2026 — Phase 2 · Team JLShen

Source code for our Phase-2 entry in the Capgemini "Prediction of Sea Winds" hackathon:
**Task 1** (probabilistic downscaled AROME wind forecast), **Task 2** (55 × IEA-22MW wind-farm
siting), and the economics behind the written report. Codabench user **junlong**.

**First place of 3 published finalist entries** on the Phase-2 final leaderboard (phase 29696),
by mean rank across the six scored sub-dimensions.

| metric | ours | runner-up | third |
|---|---|---|---|
| **mean rank (ranking column)** | **1.167** | 2.000 | 2.833 |
| speed d+1 / d+7 / d+14 | **8.98** / **17.60** / **15.41** | 9.73 / 24.83 / 16.69 | 40.63 / 34.08 / 43.65 |
| direction d+1 / d+7 / d+14 | **66.42** / 319.13 / **333.60** | 83.09 / **313.66** / 343.44 | 1061.83 / 348.63 / 334.09 |

Scores are the organisers' own per-dimension metrics (lower is better) on the withheld 2022
inference set; ranking is the mean of the six per-dimension ranks, so every sub-dimension carries
equal weight. We lead five of six and are second at direction d+7.

Released under the **MIT Licence** (see `LICENSE`), as required by the Hackathon Rules §7.

## Method in brief

- **Speed** — LightGBM quantile MOS on ECMWF-HRES at d+1, with a climatology floor at d+7/d+14,
  calibrated by conformalised quantile regression and a coverage-targeted interval width.
- **Direction** — a self-generated 8-member Pangu-Weather IC ensemble driven by ERA5, whose
  ensemble-mean centre is the single largest gain in the campaign; at d+14 the foundation-model
  centre measured *below* the calibrated no-skill floor, so it is switched off in favour of a
  monthly climatology centre with a solved arc half-width.
- **Siting** — a quality-diversity (MAP-Elites) layout search over the PyWake wake model with the
  real IEA-22MW power/thrust curve, screened on bathymetry and legality.
- **Economics** — bottom-up CAPEX/OPEX, a day-ahead market simulation with quantile bidding, and
  a forecast-value estimate, all computed rather than asserted.

`REPRODUCE.md` is the authoritative, command-by-command reproduction path for **every number in
the report**, including the exact operating points of the submitted entry (§2.1).

## Layout

| path | contents |
|---|---|
| `scripts/` | 35 Python modules — forecast pipeline, siting search, market/economics model, acceptance gates |
| `scripts/results/` | the small measured input artifacts the code reads (market-value runs, constructibility, compute footprint, layout search) |
| `REPRODUCE.md` | stage-by-stage rebuild instructions and verification commands |

The forecast entry is regenerated end-to-end by `scripts/p2_rebuild_predictions.py`, which runs in
stages (`plan → era5 → ens → pangu → assemble`). On the withheld 2022 inference set the full chain
ran in 48 minutes unattended. `--stage verify` byte-compares a rebuild against a reference
artifact, and `--stage assemble --no-fm` produces a schema-legal fallback in ~96 s with no
foundation-model dependency.

## Environment and paths

Python 3.11; see `requirements.txt`. `scipy` is supplied by the cluster module stack rather than
pip in our environment, so it is listed unpinned.

Modules resolve the project root from the `SEAWINDS_ROOT` environment variable, defaulting to this
directory. Set `SEAWINDS_ROOT` to wherever you unpack it, plus `SEAWINDS_PHASE2_DIR` and
`SEAWINDS_DATA_DIR` for the Phase-2 and Phase-1 data trees.

Intermediate and output artifacts (ERA5 initial conditions, Pangu rollouts, footprint caches,
candidate `predictions.csv`) default to a relative `./work/` directory. Each is also overridable
by its own environment variable, named at the point of use — `SEAWINDS_FM_DIR`,
`FM_ERA5_INIT_DIR`, `P2_BASE_OUTDIR` and the `--out` / `--work` flags. Point them at fast local
storage: a full rebuild writes tens of GB.

## Validation protocol

Leave-window-out / temporal-blocked cross-validation over 2016–2020, mirroring the evaluation
(14 days of context → d+1/d+7/d+14 at the footprint), scored with the exact speed-Winkler and
circular-Winkler metrics — no AROME leak and no cross-window context leak. Operating points
calibrated on 2021 were all re-read on the withheld 2022 set before shipping; three inverted and
were changed. `scripts/seawinds_metric.py` is our own reimplementation of the organisers' scoring
metric, used as the local iteration signal.

## What is deliberately NOT here, and why

- **The fitted artifacts** (`models_final/`: quantile MOS bundles, footprint climatology, ensemble
  fits, canonical footprint ordering). They are trained on — and in two cases directly derived
  from — the organisers' competition data, which is not ours to redistribute. `REPRODUCE.md` §1
  names the command that regenerates each of the six from the organisers' own data, and they are
  available to the organisers on request.
- **The Pangu-Weather ONNX checkpoint.** Licensed **CC BY-NC-SA 4.0**; the upstream repository
  states that commercial use of the models is forbidden. Redistributing it inside an MIT release
  would be a contradiction, so `scripts/fm_pangu_infer.py` downloads it from the upstream release
  instead. **Nothing in this directory carries a non-commercial restriction.**
- **The organisers' own starting kit** — `forecast_hres.py`, `cost_model.py` and the PyWake
  wind-farm simulator wrapper. That is their code, not ours to relicense; our modules import it
  from the `phase_2` branch of the official Hackathon repository. Place the kit on `PYTHONPATH`,
  or set `PHASE2_DATA_ROOT` as `REPRODUCE.md` describes.
- **The competition data.** Request it from the organisers.

## Provenance notes

- Every reported result traces to an on-disk artifact; `REPRODUCE.md` names the command that
  produced each one.
- AROME is used **only as a training target**, never as a model input, at any resolution. The
  ERA5/Pangu ensemble is initialised at each window's own `context_end` with no data past the
  issue time.
- The ENTSO-E / energy-charts price data feeds the report's economics **only** — never training,
  never feature engineering, and never either submitted deliverable.
