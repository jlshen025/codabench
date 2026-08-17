# UDIVA-HHOI Track 4 — Multimodal Egocentric Event Anticipation

Team **JLShen** (Codabench user `junlong`) · Competition 16645 · Test phase 27271
Final submission: **825333** — server subtask scores: next-action **0.4641**,
verbal **0.6548**, non-verbal **0.5301**, verbal+non-verbal **0.3872**.

Complete code to reproduce the submission from the development annotations, plus the
comparative experiments behind the negative result.

## Method

For every (segment, participant) cell the predictor outputs the same K=5 alternative
hypothesis sequences, selected once from the 21 annotated development sessions. A candidate
pool of 147 sequences (verbal and non-verbal modal singletons, mixed and repeated pairs,
observed training sequences, and the empty sequence) is built from the training event
statistics, and a greedy best-of-K set optimizer (`GreedyK5Predictor`) picks the 5-set
maximizing the summed best-of-K normalized SDL score over the training cells and the four
subtask reductions, weighting the four subtasks equally. The selected pack is frozen in
`code/heldbest_hedgepack.json` and applied identically to every cell; the released test
template is filled in place. Because the metric keeps the best alternative per subtask
independently, one diverse 5-set serves all four columns. Cross-validated development found
no observed-prefix model (ego video, audio, transcript) whose gain over this prior was
stable under implementation choices and replicated on held-out data at the 2-second horizon,
so the jointly-optimized hedge-pack is the shipped solution. At inference no test-segment
input is read, and no fatigue/mood metadata is used anywhere.

The track fact sheet submitted to the challenge organizers documents the objective, the pool
construction, the validation protocol and the full comparison tables.

## Environments

| file | purpose |
|---|---|
| `requirements-core.txt` | **the submitted method** — Python ≥3.8 + numpy; no GPU, no pretrained model |
| `requirements-experiments.txt` | the comparative experiments — torch, timm, decord, transformers, librosa, scikit-learn (+ an `ffmpeg` binary, GPU recommended) |
| `requirements-frozen-cluster.txt` | exact frozen cluster venv of the original runs; provenance record only, not portable |

```bash
python -m venv venv && . venv/bin/activate && pip install -r requirements-core.txt
export UDIVA_ROOT=/path/to/UDIVA-HHOI     # the directory holding development/ and evaluation/
```

`UDIVA_ROOT` is the only path you need to set for the submitted method (`code/udiva/io.py`
derives the rest). The experiments additionally use `UDIVA_FEAT_DIR` and `UDIVA_TEXT_DIR`.

## Reproduce submission 825333

```bash
cd code
python run_hedge2.py           # rebuild the pack -> writes heldbest_hedgepack.json  (~20 min)
python make_test_sub_fixed.py --out anticipation_submission.zip   # fill the official template
python run_report.py           # every number quoted in the fact sheet -> report.json (~15 min)
```

`run_hedge2.py` runs the pool hyper-parameter sweep, the held-out check on the organizer
development grid, the per-fold consistency check, the final fit on all 21 sessions, and the
hill-climb local-optimality certificate — then writes the pack. Pass
`--config nv12_nnv10_mix5` to pin the selected configuration and skip the sweep.

Everything is deterministic (no seeds, no RNG). The rebuilt `heldbest_hedgepack.json` is
byte-identical to the packaged one, and the `anticipation.json` built from it is
byte-identical to the file inside submission 825333:

```
sha256(anticipation.json) = ea43f9e899c59c3ab00b1b83461922fb919f6c3dc759079d1d55f41ce21a80a2
```

## Layout

| step | script |
|---|---|
| parsing, grids, CV, metric | `code/udiva/` package |
| candidate pool + greedy K=5 optimizer | `code/udiva/baselines.py` |
| pack build, validation, local-optimality check | `code/run_hedge2.py` |
| full fact-sheet report (pool anatomy, LOSO, dev grid, oracle) | `code/run_report.py` → `report.json` |
| our own output of that report, for comparison | `code/report_reference.json` |
| packer (fills the official template) | `code/make_test_sub_fixed.py` |
| rejected transcript-conditioned model | `code/run_devval.py`, `code/udiva/models.py` |
| comparative experiments + feature extraction | `code/experiments/` (see its README) |

## Reproducibility of the comparisons

The DINOv2 and bge-m3 feature caches of the original runs were stored on an automatically
purged scratch filesystem and no longer exist, so the numbers reported for those experiments
come from the preserved run logs. `code/experiments/extract_features.py` and
`code/experiments/embed_transcripts.py` regenerate both caches (about one GPU-hour) with the
exact checkpoints and settings used. `code/experiments/crossview_scout.py` re-implements the
recorded cross-view protocol, whose original one-off script was not kept.
Everything concerning the submitted system is fully reproducible and was re-executed while
preparing this revision.

## License

MIT (see `LICENSE`); the code remains publicly accessible for at least three years.
