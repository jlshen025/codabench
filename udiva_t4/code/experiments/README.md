# Comparative development experiments (all rejected)

These scripts produced the negative results that justify the input-free submission. **None
of them is needed to reproduce the submission** — see the top-level `README.md` for that.
Install `requirements-experiments.txt` (not `requirements-core.txt`) and run everything
from the parent `code/` directory, e.g. `python experiments/diag_features.py`.

Paths come from environment variables:

| variable | meaning | default |
|---|---|---|
| `UDIVA_ROOT` | unpacked challenge data (holds `development/`, `evaluation/`) | the cluster path |
| `UDIVA_FEAT_DIR` | DINOv2 feature cache | `./features` |
| `UDIVA_TEXT_DIR` | bge-m3 transcript-embedding cache | `./text_emb` |
| `EXP_OUTPUT_DIR` | where a script writes its `result.json` | `.` |

## Feature caches (run these first)

```bash
export UDIVA_ROOT=/path/to/UDIVA-HHOI UDIVA_FEAT_DIR=$PWD/features UDIVA_TEXT_DIR=$PWD/text_emb
for i in $(seq 0 20); do SLURM_ARRAY_TASK_ID=$i python experiments/extract_features.py; done
python experiments/embed_transcripts.py
```

* `extract_features.py` — frozen **DINOv2 ViT-B/14**, timm id `vit_base_patch14_dinov2.lvd142m`
  (the `facebook/dinov2-base` LVD-142M weights), `num_classes=0` so the output is the 768-d
  CLS embedding, `img_size=224`. Both ego views (E1 → participant A, E2 → participant B) are
  decoded with decord at **4 fps**, resized to 224×224 by the decoder, normalized with the
  model's own `timm` data config, encoded in batches of 96 under `torch.no_grad()`, and stored
  as fp16 per session together with the frame timestamps (~1.9 GB total for 21 sessions).
* `embed_transcripts.py` — **`BAAI/bge-m3`** (`AutoModel`, CLS token, L2-normalized, max 128
  tokens), one 1024-d embedding per transcript utterance, with utterance start/end times and
  speaker ids.

## Experiments

| script | approach | conditioning input |
|---|---|---|
| `diag_features.py` | linear probe on DINOv2 own-view appearance | 2 s window, `mean` pooling |
| `run_knn_cv.py` | feature-neighbour retrieval (`FeatureKNN`) + static-seeded hybrid | 1.0/2.0 s windows, `mean` and `mean_last` pooling |
| `motion_scout.py` | motion proxies derived from the DINOv2 window (endpoint difference, per-dim std, mean absolute frame-to-frame change) | 2 s window |
| `crossview_scout.py` | cross-view prediction (partner's ego view → this participant's next action) | 2 s window, `mean_last` |
| `audio_scout.py` | handcrafted librosa audio features | 2 s window, own + partner ego audio |
| `transcript_diag.py` | bge-m3 transcript probe (verbal-active, next utterance type) | 4 s window |
| `run_cond.py` | conditional verbal hedge, `n_types` × window sweep | 4 s / 8 s windows |
| `run_dialogue.py` | dialogue-structure features vs bge-m3 vs both | 6 s window |
| `run_phase.py` | session-phase-conditioned hedge-packs (2/3/4 buckets) | time in session only |

`../run_devval.py` scores the best conditional model on both selection protocols, and
`../run_report.py` (section 6) computes the verbal oracle.

The probes share one protocol: the anchored grid, a group-held-out split (every 4th session
held out), multinomial logistic regression (`C=0.5`, scikit-learn), and top-1/top-5 accuracy
against the training-frequency prior. The end-to-end predictors are scored by anchored
leave-one-session-out CV with the four-subtask SDL metric.

## Reproducibility status

* The scripts, checkpoints, windows and hyper-parameters are complete, so every experiment
  can be re-run from the challenge data.
* The **cached features themselves are gone** — they lived on an auto-purged scratch
  filesystem — so the numbers quoted in the fact sheet come from the preserved run logs
  rather than from a re-execution. Re-running `extract_features.py` + `embed_transcripts.py`
  regenerates them (about one GPU-hour in total).
* `crossview_scout.py` re-implements the recorded cross-view protocol; that particular run's
  original one-off script was not kept.
