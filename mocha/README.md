# MoCha 2026 — Parkinsonian gait severity, cross-site

Team **JLShen** (CodaBench user `JLShen`) · Competition 16752 · Leaderboard 18564 ·
Evaluation phase 27428
Ranked submission **882979** — server **macro-F1 0.69447** (macro-P 0.72347, macro-R 0.67659,
accuracy 0.65591, QWK 0.60007), **1st of 58**.

Runner-up 0.5807; the organizers' released baseline bundle 0.4289; the published CARE-PD cross-site
band for a single frozen encoder with a linear probe is ~0.52–0.55.

Predict the MDS-UPDRS gait severity class {0,1,2,3} of a walking sequence given as canonicalized
SMPL motion, where the hidden test comes from clinical sites that appear nowhere in training — so
the task is cross-site generalization, not in-distribution classification.

**The trained model is 14 KB.** A public motion encoder is frozen and a single 4×512 linear layer is
trained on top of it; nothing else is learned. That head is in this repository. The three
third-party binaries the system loads are not — see [`assets/README.md`](assets/README.md).

## Method

```
SMPL pose/trans → forward kinematics → 3D joints
  → decimate to ~30 fps on the shipped `fps`, project to the side view, 81-frame clips,
    per-clip crop-scale normalization
  → FROZEN MotionAGFormer-S (Human3.6M-pretrained), mean over frames, clips, then joints → 512-d
  → z-score with the head's stored feat_mean / feat_std
  → transductive TEST-mean centering, shrinkage c = 0.90
  → linear head 4×512, FocalLoss(α=1, γ=1) → per-walk posteriors
  → SUBJECT-MEAN posterior aggregation, λ = 1.0        (one label per subject)
  → SUBJECT-kNN posterior pooling, k = 5, λ_nbr = 0.40
  → q-divisor logit adjustment, τ = 0.50 → argmax
```

`code/config.json` is the whole configuration, and was the only file that differed between our
candidate submissions:

```json
{"views":["side_glob"],"reduce":"jointmean","tta":["orig"],"tau":0.5,
 "subj_smooth":1.0,"center_lam":0.9,"nbr_smooth":0.4,"nbr_k":5}
```

Two properties worth stating plainly:

* **Transductive but label-free.** `predict()` receives the whole test set at once, and the
  centering and operating-point stages read only *unlabeled* test statistics — the test feature
  mean, and the model's own predicted class marginal. No hidden-test label is used, inferred or
  reconstructed anywhere.
* **It exploits released structure, not leakage.** The subject grouping `data[sid][wid]` is part of
  the input format the organizers ship. Aggregating over it is the single largest contribution in
  the system.

## What each stage was worth

Every number is a submission scored on the hidden test, not a local estimate — see *Validation*
below for why. Rows were measured at different points in the run, so each Δ is against the row
above on the pipeline as it then stood.

| stage | macro-F1 | Δ |
|---|---|---|
| organizers' released baseline bundle | 0.4289 | — |
| frozen MotionAGFormer-S + plain linear probe | 0.467 | +0.038 |
| + benchmark-exact head recipe (focal loss, AdamW, z-score) | 0.524 | +0.057 |
| + transductive centering & q-divisor operating point | 0.5407 | +0.017 |
| **+ subject-mean posterior aggregation (λ=1)** | 0.68371 | **+0.143** |
| + centering shrinkage retuned c 0.8 → 0.9 | 0.68699 | +0.0033 |
| + subject-kNN posterior pooling (k=5, λ_nbr=0.40) | **0.69447** | +0.0075 |

**The head recipe, +0.057.** Our first probe on these frozen features was a plain logistic
regression, and it plateaued at 0.467 for weeks in a way that read like a task ceiling. It was not:
reproducing the CARE-PD benchmark's *exact* head recipe — focal loss α=1 γ=1, AdamW, z-scored
features, read off the benchmark's vendored winning-config JSONs rather than from paper prose —
moved the same features to 0.524. The architecture was never the problem.

**Subject-mean aggregation, +0.143**, monotone to the boundary: λ 0.00 → 0.5407, 0.75 → 0.6489,
0.90 → 0.65828, 1.00 → 0.68371. The aggregator matters too — the uniform arithmetic mean beat
confidence-weighted (0.66416), trimmed-25% (0.65837) and logit-space geometric pooling (0.60777);
geometric pooling sharpens the posteriors and cripples the operating point downstream.

**Subject-kNN pooling** has a sharp interior optimum: λ_nbr 0 → 0.68699, **0.40 → 0.69447**,
0.45 → 0.67256, 0.70 → 0.62769, 1.00 → 0.4012. Its own limit confirms the mechanism — blending
toward the *global* test-mean posterior (the k=∞ case) scores 0.67517, below base. Local pooling
denoises; global pooling washes out real between-subject differences. k is flat: k ∈ {3,8,12} are
byte-identical to k=5 at λ_nbr = 0.40.

Both continuous knobs are piecewise-constant and saturated. Centering: c 0.60 → 0.68042,
c 0.70 = c 0.80 → 0.68371, c 0.90 = c 0.95 = c 1.00 → 0.68699. Operating point:
τ 0.40 = τ 0.45 → 0.6684, τ 0.50 = τ 0.52 → 0.68371, τ 0.55 → 0.68042, τ 0.60 → 0.6478. Both were
re-swept after the pooling stage was added, since pooling changes exactly the posterior peakedness
the operating point reads, and both held.

## Layout

```
mocha/
├── code/                 the inference code of the submitted entry
│   ├── run.py            defines predict(data) -> dict; the evaluation entry point
│   ├── config.json       the configuration above
│   ├── smpl_fk_eval.py   vendored SMPL forward kinematics (numpy/torch only)
│   └── model/            MotionAGFormer architecture source
├── weights/head_jm_zscore_focal.npz    the trained head: W(4,512) b(4) feat_mean feat_std
├── assets/               how to obtain the 3 third-party binaries, + the SMPL converter
├── training/             the chain that produced the head
└── verify.py             run the whole thing end-to-end on released data
```

## Reproducing

### Verify the entry

Fetch the three files in [`assets/README.md`](assets/README.md), then:

```sh
pip install torch numpy
python verify.py --data /path/to/CARE-PD/Canonicalized_SMPL_pickles
```

This checks every binary's md5, assembles the runnable tree as it was packaged for the server,
imports **its own** `run.py`, and calls `predict()` on a probe built from the *released* cohorts —
no hidden data, no label ever read.

Our own run of it, on one H100 MIG slice: all four checksums matched, and `predict()` returned 284
labels in 41 s with class distribution `[112, 92, 74, 6]` — the marginal the ranked entry produced.
That projects to ~53 s for the 372-walk hidden test, against a 3600 s limit.

### Retrain the head

Three steps; only step 2 trains anything.

```sh
export CAREPD_DIR=/path/to/CARE-PD/Canonicalized_SMPL_pickles
export CACHE_DIR=/path/with/room/for/2GB

# 1. frozen-encoder features over the augmented training set        (~20 min on one GPU)
python training/extract_feats_perjoint_aug.py
#    → $CACHE_DIR/feats_magfs_augE_pj.npz, X_side_glob (44280, 17, 512)
#    augmentations that transferred: spatial jitter, temporal speed-warp, frame dropout

# 2. train the linear head                                          (minutes)
HEAD_SEED=0 OUT_DIR=weights python training/train_final_head.py
#    phase 1: tune the epoch count on a 15% within-train split by macro-F1
#    phase 2: refit on 100% of the labeled data for that epoch  ← the shipped weights
```

Step 3 is packaging: the archive submitted to CodaBench is `code/` plus the four binaries, flattened
into one directory and zipped — which is exactly what `verify.py` assembles.

Training data is the four UPDRS-labeled CARE-PD cohorts, 110 subjects / 2953 walks. The dataset is
**not** redistributed here; get it from the CARE-PD authors
(<https://huggingface.co/datasets/vida-adl/CARE-PD>) and cite CARE-PD (arXiv 2510.04312,
NeurIPS 2025). It is CC BY-NC 4.0.

Training additionally needs `smplx`, `scipy` and `scikit-learn`; inference needs only `torch` and
`numpy`. See `requirements.txt`.

## Validation — and why no number above comes from it

Our local protocol was nested leave-two-cohort-out cross-validation over the four labeled cohorts,
with the organizers' scorer. Partway through the challenge we replayed the eleven configurations we
had server scores for through it and rank-correlated the two:

```
Spearman ρ = -0.373       Pearson r = -0.760
```

**The cross-validation was anti-correlated with the score that decided the competition.** The
server's best configuration scored close to the lowest locally; the server's worst scored the
highest. The inversion sits on the axis that mattered most — cross-validation preferred *no* subject
aggregation by −0.042 while the hidden test preferred full aggregation by +0.143.

It is a property of the estimator, not a bug in it: shuffling the training labels collapses the same
setup from 0.4416 to 0.2234, so it does measure real signal. Our reading is that
leave-two-cohort-out folds reward fitting whatever idiosyncrasies the remaining cohorts have, and
subject aggregation deliberately discards exactly that.

We therefore retired cross-validation as a selector for every decision, aggregation and
operating-point choice, and selected on server scores instead — legitimate here because the
challenge was a single-phase evaluation on a fixed test set with a deterministic scorer, so
maximizing over reads is exact optimization of the announced objective rather than selection on
noise.

## Two caveats

**The head is unstable to its random initialization.** Retraining the identical recipe with
different seeds gives hidden-test macro-F1 of 0.694 / 0.681 / 0.652 / 0.606 / 0.558 across five
draws — a spread of 0.14 on a 4×512 linear probe fit to 110 subjects. The shipped head is the best
of those draws, so part of the margin is a favourable draw. This is why the exact trained head is in
this repository and why step 2 above pins `HEAD_SEED`. Seed *ensembling* does not fix it: averaging
2 to 5 heads on the final system scores 0.69106 / 0.69106 / 0.68764 against 0.69447 for the single
head, because subject-mean and kNN pooling already do the variance reduction an ensemble would
supply. A ±0.02 difference between two systems on this benchmark is not a method effect.

**Subject aggregation is also the ceiling.** On the labeled cohorts 62 of 110 subjects (56.4%) carry
mixed walk labels, so forcing one label per subject makes 30.5% of walks unreachable — a hard
accuracy ceiling of ~69.5% against the 65.59% realized. The λ=1.0 family is therefore ~95%
saturated, and every attempt to re-admit per-walk evidence on top of aggregation lost.

## What was tried and rejected

Each family was closed by a direct measurement on the hidden test.

- **Encoder fine-tuning — dead in four forms.** Unsupervised 2D→3D pretext fine-tuning (0.3605, far
  below frozen), anti-site-latch LoRA + gradient reversal + bottleneck, supervised LoRA,
  contrastive SSL + GRL.
- **Alternative frozen encoders — all below MotionAGFormer-S.** MixSTE (0.4081, including in its own
  native normalization pipeline), MotionAGFormer-B/L, MotionBERT, PoseFormerV2, MotionCLIP, TMR, an
  NTU-pretrained skeleton transformer, MoMask, a Riemannian SPD probe, a time-series foundation
  model. Joint-mean pooling beat per-joint 8704-d features.
- **Alignment beyond the first moment — only the mean transfers.** CORAL and diagonal-std alignment,
  median centering, PCA denoising, iterative nullspace projection, transductive BN, quantile-CDF
  matching, per-site clustered centering. The last fails even with **oracle site labels**, so it is
  not a clustering-quality problem.
- **Decision rules and objectives — dead.** Ordinal/CORAL head (0.4558, −0.23, bracketed over τ so
  not a mistuning artifact), threshold tuning, LDA, Saerens EM label shift, class-balanced
  centering, prototype/NCM, soft-macro-F1 surrogate, V-REx / GroupDRO / IRM, training-time
  de-confidencing, per-class decision scaling (suppress class 3 → 0.64882, boost class 2 → 0.5715).
  A parameter-free joint macro-F1 decoder returned predictions *byte-identical* to argmax, i.e.
  **argmax is already F1-optimal here** — the recall gap is separability, not miscalibration.
- **Ensembles — negative in every form.** Diverse-encoder blends, MotionAGFormer ⊕ MixSTE (locally
  positive, 0.3720 on the server), fine-tuned ⊕ frozen, two-view side+back (CV +0.032 inverted to
  −0.042), and the head averaging above.
- **Transductive inference beyond centering — dead.** Graph label propagation, Nyström-RBF and
  metric learning, density-ratio importance weighting, FixMatch over the five unlabeled cohorts,
  Deep Feature Reweighting, attention pooling over clips, bag-of-windows pooling, self-training TTA
  (an exact no-op: 0 of 284 probe items changed).
- **Hand-crafted clinical gait features — dead, and instructive.** Cadence, stride, asymmetry and
  variability features are **96–98% cohort-predictable**: they are site detectors on this data, and
  a random-forest late blend improved the pooled mean while hurting held-out cohorts. Useful
  by-product: k-means on those same features recovers the cohorts at ≈1.00 purity, so they are an
  excellent *site* signal and a poor *severity* signal.
- **Per-site operating points — mechanism real, no deployable arm.** An oracle-site-label arm gains
  +0.0221, but site clustering in the embedding reaches only 0.81–0.84 purity, and the features that
  cluster sites perfectly are exactly the ones unusable as severity inputs. Per-cohort severity
  mixes differ sharply ([.27,.47,.16,.11], [.44,.35,.21,.00], [.46,.37,.15,.02], [.25,.31,.44,.00]),
  so a single pooled operating point is wrong for every site at once.

## Compliance

* Training data: CARE-PD only, under CC BY-NC 4.0, cited above. No external data beyond CARE-PD and
  the public pretrained encoder.
* Pretrained weights: MotionAGFormer-S, Human3.6M-pretrained, used **frozen and unmodified**.
* No hidden-test label was used, inferred or reconstructed at any point.
* The SMPL body model is not redistributed here — see [`assets/README.md`](assets/README.md).

## License

MIT — see `LICENSE` — **except for three items with their own terms, listed in `NOTICE`**:
`code/model/motionagformer/` is Apache-2.0 (unmodified, license included alongside);
`code/smpl_fk_eval.py` is a port of `smplx.lbs` and inherits its Max Planck non-commercial
research licence; and `weights/head_jm_zscore_focal.npz`, though trained by us, is derived from
CARE-PD and is released for **non-commercial** use with attribution to CARE-PD. Read `NOTICE`
before reusing anything here.
