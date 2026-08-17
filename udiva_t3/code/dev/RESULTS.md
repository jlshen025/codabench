# Development experiments — complete results

**None of the approaches on this page is part of the final CodaBench submission.** The
submitted entry (825331) is the constant K=5 hypothesis set described in `../../README.md`
§1 (`predictors.b2_factory`). Everything here was implemented, cross-validated and
**rejected** during development; it is included so that the comparison reported in the fact
sheet can be checked and reproduced.

Every number below was **recomputed with the code in this directory under one single
protocol** — the leave-one-session-out protocol of `../../README.md` §5 (21 folds, 2835
segments = 5670 participant cells, `sdl.py` scorer, 4 subtask columns, `mean` = unweighted
mean of the four). Columns: `next` / `verbal` / `nonverbal` / `mixed` (the metric's four
subtask reductions) and `mean`.

**Noise floor.** The fold-to-fold standard deviation of the mean is **0.039** for the
submitted predictor (`fstd` column of `python ../cv.py`). With 21 sessions, differences below
that are not resolvable. `cv.py`-based rows print their own `foldstd`; the retrieval/probe
rows (`vidprobe.py`, `vidtrain.py`) pool folds without recomputing it, so compare them
against the same 0.039.

**Bit-level reproducibility.** The input-free rows (reference rows, §1, §2, §4, §5) use only
the Python standard library and are reproducible exactly. The retrieval and probe rows of §3
can move in the **fourth decimal** across machines and thread counts: the cosine similarities
come from a threaded BLAS matrix product whose summation order decides which of several
near-equally-similar neighbours is retrieved. Measured spread over
`OMP_NUM_THREADS` ∈ {1, 4, 8}: ≤ 0.0001 per column, means unchanged (e.g. transcript k-NN
k=20 mixed column 0.2808/0.2809, mean 0.3815 in all runs).

**Reference rows** (`python ../cv.py`):

| approach | next | verbal | nonverbal | mixed | mean |
|---|---|---|---|---|---|
| all-empty predictions (floor) | 0.0628 | 0.4975 | 0.1093 | 0.0628 | 0.1831 |
| **B2 — SUBMITTED (825331)** | 0.3902 | 0.5760 | 0.3806 | 0.3041 | **0.4127** |
| B4 — non-verbal-heavy A/B variant (837289) | 0.4114 | 0.4975 | 0.4261 | 0.3185 | 0.4134 |

---

## 1. Event-history Markov models (oracle) — `python cond.py`

Context read from the **ground-truth** annotations of the observed past, i.e. as if a perfect
recognizer existed; this bounds every recognizer-based version of the same idea. Table
P(future-window sequence | context) counted on the training fold; prediction = empty sequence
+ the top-3 sequences of the matching context row + backfill with the submitted B2
alternatives up to K=5 (so the conditioned model can only differ from B2 where its own
predictions fire). `W` = lookback window.

| context | next | verbal | nonverbal | mixed | mean |
|---|---|---|---|---|---|
| last event, W=2 s | 0.4066 | 0.5370 | 0.4172 | 0.3154 | 0.4190 |
| **last event, W=4 s** (best of all conditioning) | 0.4193 | 0.5399 | 0.4217 | 0.3200 | **0.4252** |
| last two events, W=2 s | 0.3901 | 0.5572 | 0.3935 | 0.3042 | 0.4112 |
| last two events, W=4 s | 0.3932 | 0.5655 | 0.3924 | 0.3057 | 0.4142 |
| bag of events, W=2 s | 0.3875 | 0.5593 | 0.3892 | 0.3034 | 0.4098 |
| bag of events, W=4 s | 0.3886 | 0.5717 | 0.3831 | 0.3031 | 0.4116 |
| last high-level action only, W=2 s | 0.3905 | 0.5100 | 0.4104 | 0.3069 | 0.4045 |
| last high-level action only, W=4 s | 0.3959 | 0.5097 | 0.4150 | 0.3111 | 0.4079 |
| last target only, W=2 s | 0.4051 | 0.5085 | 0.4181 | 0.3126 | 0.4111 |
| last target only, W=4 s | 0.4186 | 0.5094 | 0.4287 | 0.3213 | 0.4195 |
| event count (activity rate), W=2 s | 0.3490 | 0.4975 | 0.3772 | 0.2790 | 0.3757 |
| event count (activity rate), W=4 s | 0.3428 | 0.4975 | 0.3733 | 0.2769 | 0.3726 |

Reading: the best oracle context gains **+0.013** over the constant prior — a third of the
noise floor — and it buys that on the non-verbal/next columns while *losing* 0.036 on the
verbal column. Coarsened contexts (high-level action, target) and activity rate are worse.
Since even the oracle is inside the noise, no recognizer of the past can turn this family into
a real gain, which is what §3 confirms directly.

## 2. Ongoing-activity conditioning (oracle) — `python cond.py`

Context = the event(s) **in progress** at the reference timestamp (`start <= t_b < end`),
again read from the ground truth; same table and prediction rule as §1.

| context | next | verbal | nonverbal | mixed | mean |
|---|---|---|---|---|---|
| exact ongoing event tuple(s) | 0.3651 | 0.5203 | 0.3843 | 0.2870 | 0.3892 |
| ongoing high-level action(s) only | 0.3559 | 0.5094 | 0.3827 | 0.2841 | 0.3830 |

Reading: knowing exactly what a participant is doing at $t_b$ is **worse** than not
conditioning at all (−0.024). What is in progress does not predict what starts in the next
2 s better than the marginal distribution does.

## 3. Frozen-representation probes on the observed prefix

### Feature extraction

* **Video** (`feat_extract.py`) — `facebook/dinov2-small` (ViT-S/14, 384-d CLS token),
  **frozen**, no fine-tuning. Exocentric GF view. Observation window **3 s ending at $t_b$**,
  **6 frames** uniformly spaced inside it (nearest decoded frame index, `decord`). Two crops
  per frame: left 60 % of the width for `participant_a`, right 60 % for `participant_b`; each
  resized to 224×224 with ImageNet normalization. Per (segment, participant) feature block
  `[6, 384]`.
* **Transcript** (`txt_extract.py`) — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (384-d, **frozen**, L2-normalized output). Text = concatenation of all `.srt` utterances
  whose start falls in the **10 s before $t_b$**; one vector per segment, shared by both
  participants.

### k-NN retrieval — `python vidprobe.py video` / `python vidprobe.py text`

Query vector = pooled prefix features (`mean`, `last` frame, or concat of both), L2
normalized; cosine similarity against every training (segment, participant) cell of the fold,
restricted to the same participant role. Alternatives = empty + the 2 most frequent
ground-truth sequences among the k neighbours + the most frequent single neighbour event +
B2 backfill up to K=5.

| features | k | pooling | next | verbal | nonverbal | mixed | mean |
|---|---|---|---|---|---|---|---|
| DINOv2 video | 10 | mean | 0.3573 | 0.5206 | 0.3884 | 0.2881 | 0.3886 |
| DINOv2 video | 20 | mean | 0.3517 | 0.5157 | 0.3818 | 0.2829 | 0.3830 |
| DINOv2 video | 40 | mean | 0.3389 | 0.5079 | 0.3711 | 0.2744 | 0.3731 |
| **DINOv2 video** | **10** | **last** | 0.3617 | 0.5256 | 0.3918 | 0.2916 | **0.3927** |
| DINOv2 video | 20 | last | 0.3534 | 0.5217 | 0.3831 | 0.2845 | 0.3857 |
| DINOv2 video | 40 | last | 0.3416 | 0.5108 | 0.3727 | 0.2766 | 0.3754 |
| DINOv2 video | 10 | mean+last | 0.3589 | 0.5225 | 0.3890 | 0.2900 | 0.3901 |
| DINOv2 video | 20 | mean+last | 0.3530 | 0.5167 | 0.3842 | 0.2852 | 0.3848 |
| DINOv2 video | 40 | mean+last | 0.3410 | 0.5082 | 0.3727 | 0.2764 | 0.3746 |
| **transcript emb.** | **10** | mean | 0.3530 | 0.5260 | 0.3863 | 0.2871 | **0.3881** |
| transcript emb. | 20 | mean | 0.3442 | 0.5216 | 0.3794 | 0.2808 | 0.3815 |
| transcript emb. | 40 | mean | 0.3288 | 0.5114 | 0.3668 | 0.2707 | 0.3694 |

### Trained linear probe — `python vidtrain.py video` / `python vidtrain.py text`

Multi-label linear layer over the `N_VOCAB=40` most frequent event tuples, trained per fold on
the pooled (`mean+last`) features: `torch.nn.Linear`, `BCEWithLogitsLoss`, full-batch Adam,
lr 0.05, weight decay 1e-2, 150 steps. Prediction: events with sigmoid probability above `thr`
(at most `maxlen=3` of them) become alternatives 2–3; alternative 1 is the empty sequence and
the rest is B2 backfill.

| features | thr | next | verbal | nonverbal | mixed | mean |
|---|---|---|---|---|---|---|
| DINOv2 video | 0.25 | 0.3532 | 0.4975 | 0.3966 | 0.2973 | 0.3861 |
| **DINOv2 video** | **0.35** | 0.3698 | 0.5349 | 0.3806 | 0.2918 | **0.3943** |
| DINOv2 video | 0.50 | 0.3902 | 0.5760 | 0.3806 | 0.3041 | 0.4127 † |
| transcript emb. | 0.25 | 0.3532 | 0.4975 | 0.3967 | 0.2974 | 0.3862 |
| **transcript emb.** | **0.35** | 0.3690 | 0.5338 | 0.3811 | 0.2918 | **0.3939** |
| transcript emb. | 0.50 | 0.3902 | 0.5760 | 0.3806 | 0.3041 | 0.4127 † |

† **Degenerate row, not a video/transcript result:** at `thr=0.50` no event ever exceeds the
threshold, so no model prediction enters the hypothesis set and the predictor reduces
*exactly* to B2 — which is why the numbers match B2 digit for digit. Whenever the probe
actually fires it loses ground (0.386–0.394 < 0.4127). Both modalities land on the same value
because the surviving alternatives are the same prior.

Reading (§3 overall): with the observed prefix encoded by a strong frozen backbone, neither
retrieval nor a trained linear decoder finds usable per-segment signal — every configuration
is **below** the input-free prior on the mean (by 0.018 in the best case, 0.043 in the worst)
and below it on the `next`, `verbal` and `mixed` columns without exception. The one column
they can improve is `nonverbal` (at most +0.016, video probe at thr=0.25), which is the same
trade the input-free B4 variant makes for free. This is the direct (implementable) counterpart
of the oracle result in §1: the past that §1 shows to be marginally informative cannot be
recovered from the observed prefix well enough to matter.

## 4. Role / time-bin / session priors — `python priors.py`

Same B2 *shape* (empty + top-2 non-verbal + top-2 verbal singletons); only the subset of
training events the frequencies are counted over changes.

| prior conditioned on | next | verbal | nonverbal | mixed | mean |
|---|---|---|---|---|---|
| nothing — pooled = **submitted B2** | 0.3902 | 0.5760 | 0.3806 | 0.3041 | **0.4127** |
| participant role (A vs B) | 0.3833 | 0.5614 | 0.3806 | 0.2985 | 0.4060 |
| time bin, 30 s (5 bins) | 0.3980 | 0.5892 | 0.3806 | 0.3069 | 0.4187 |
| time bin, 60 s (5 bins) | 0.3865 | 0.5686 | 0.3806 | 0.3013 | 0.4093 |
| the session itself — **ORACLE** | 0.4002 | 0.6018 | 0.3806 | 0.3099 | 0.4231 |

Reading: per-role conditioning *hurts* — the two roles have near-identical marginals
(`participant_a`: `inspect_check/look_at/leaflet` 937, `search/look_at/available` 414;
`participant_b`: 932 / 503), so splitting the counts only thins them. The 30 s time-bin
variant gains +0.006 and the 60 s variant loses 0.003, i.e. the sign of the effect flips with
an arbitrary hyperparameter well inside the noise floor — not a usable lever. The last row is
an **oracle**: it counts the frequencies on the held-out session itself, which is the upper
bound for *any* per-session adaptation (metadata-conditioned, difficulty-conditioned, online
adaptive, …). It gains +0.010, a quarter of the noise floor, so the whole per-session-prior
family is capped far below anything worth shipping.

## 5. Alternative-set variants (longer / mixed hypotheses) — `python altsets.py`

Constant predictors that vary only the *composition* of the K=5 set (content rebuilt from each
fold's training frequencies). `NVi`/`Vi` = i-th most frequent non-verbal / verbal tuple;
`S*`/`SNV*` = most frequent *ordered* ground-truth window sequences of length ≥ 2, mined from
the training fold.

| K=5 composition | next | verbal | nonverbal | mixed | mean |
|---|---|---|---|---|---|
| **∅, NV1, NV2, V1, V2 — SUBMITTED (B2)** | 0.3902 | 0.5760 | 0.3806 | 0.3041 | **0.4127** |
| ∅, NV1..NV4 (B4) | 0.4114 | 0.4975 | 0.4261 | 0.3185 | 0.4134 |
| NV1, NV2, NV3, V1, V2 (no empty hypothesis) | 0.3389 | 0.5760 | 0.3896 | 0.2490 | 0.3884 |
| ∅, NV1, V1, [NV1,V1], [NV1,NV2] (1 mixed alt) | 0.3123 | 0.5349 | 0.3810 | 0.2988 | 0.3817 |
| ∅, [NV1,V1], [NV1,NV2,V1], [V1,V2], NV1 (2 mixed alts) | 0.3123 | 0.5552 | 0.3810 | 0.2858 | 0.3836 |
| ∅, [NV1,NV2], [NV1,NV2,NV3], [V1,V2], NV1 (longer) | 0.3123 | 0.5386 | 0.3900 | 0.3066 | 0.3869 |
| ∅, S1, S2, S3, NV1 (mined ordered sequences) | 0.3998 | 0.4975 | 0.4300 | 0.3298 | 0.4143 |
| ∅, SNV1, SNV2, NV1, V1 (mined non-verbal sequences) | 0.3698 | 0.5349 | 0.4023 | 0.3144 | 0.4054 |

Reading: spending the slot of the empty hypothesis on a third non-verbal tuple instead costs
0.024 (row 3), so the empty hypothesis stays. Multi-event and mixed
hypotheses cost the `next` column heavily (a length-≥2 alternative can never be the best
alternative for the single-event subtask) and buy at most +0.002 on the mean — inside the
noise — while tanking the verbal column, which the official per-column rank average rewards.
B2 was selected as the balanced set: no column sacrificed, verbal strongest of all variants.
`tune.py` (greedy) and `tune3.py` (exhaustive search over a curated ~12-element pool) are the
searches this choice came out of. `explore.py` prints the landscape that motivated the shape,
measured on the 254 participant cells of the two official grids with frequencies fit on the
other 19 sessions: sequence lengths per cell are mostly 2–4 events, and the empty rate is
5.9 % on the mixed column, 44.9 % on the verbal column and 9.1 % on the non-verbal column —
which is why an explicit empty hypothesis pays, most of all on the verbal subtask.

---

## Reproducing this page

```bash
export UDIVA_HHOI_ROOT=/path/to/UDIVA-HHOI      # decrypted dataset root
cd code

python cv.py                   # reference rows                      (~2 s, stdlib only)
python dev/cond.py             # §1 + §2                             (~20 s, stdlib only)
python dev/priors.py           # §4                                  (~5 s,  stdlib only)
python dev/altsets.py          # §5                                  (~10 s, stdlib only)

# §3 needs requirements-dev.txt; extraction needs the videos + a GPU (~10 min), the
# probes then run on CPU. Features are cached under $UDIVA_T3_WORK.
python dev/feat_extract.py all # DINOv2 features   -> $UDIVA_T3_WORK/feats
python dev/txt_extract.py      # transcript emb.   -> $UDIVA_T3_WORK/txtfeats
python dev/vidprobe.py video ; python dev/vidprobe.py text
python dev/vidtrain.py video ; python dev/vidtrain.py text
```

The `dev/` scripts import the shipped modules through `dev/_path.py`, so they can be run from
either `code/` or `code/dev/`.
