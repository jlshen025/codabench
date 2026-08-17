# UDIVA-HHOI Track 3 — Multimodal Exocentric Event Anticipation

Team **JLShen** (Codabench user `junlong`) · Competition 16646 · Test phase 27273
Final submission: **825331** — server subtask scores: next-action **0.375**, verbal **0.635**,
non-verbal **0.434**, verbal+non-verbal **0.344** (mean of the four columns 0.447).
(Submission 837289 was an A/B variant of the same family, `b4_factory`; 825331 is the
team's final entry.)

Complete code to reproduce the submission and every reported number from the raw data.

---

## 1. The submitted method in one paragraph

For **every** (segment, participant) cell the predictor outputs the **same** K=5 alternative
hypothesis sequences — one empty sequence and four singleton sequences, taken from the event
frequencies of the 21 annotated development sessions:

| # | hypothesis | event tuple | type | count in the 21 sessions |
|---|---|---|---|---|
| 1 | `[]` | — (empty sequence) | — | — |
| 2 | `[NV1]` | `("inspect_check", "look_at", "leaflet")` | non-verbal | 1869 |
| 3 | `[NV2]` | `("search", "look_at", "available")` | non-verbal | 917 |
| 4 | `[V1]` | `("positive_acknowledgement", "partial_model")` | verbal | 87 |
| 5 | `[V2]` | `("other_acknowledgement", "model")` | verbal | 86 |

The official normalized-SDL metric keeps the best-scoring of the K alternatives **per subtask
independently**, so this single diverse set serves all four subtask columns at once and the
empty hypothesis covers event-free horizons. **No test-segment input is read at inference** —
the predictor is a pure function of the training annotations, so only past information is
used. Cross-validated development (Section 6) found the input-conditioned alternatives at or
within the noise floor of this constant prior, which is why the constant prior was shipped.

How the frequencies are computed (`code/predictors.py:_freqs`, `b2_factory`):

* counted over the **raw annotation files** of the training sessions
  (`annotations/<sid>.json`) — i.e. over **all annotated events**, *not* over evaluation
  windows, and with no time filter;
* **pooled across all training sessions and across both participants**
  (`participant_a` + `participant_b`; `SUPERVISOR` events are ignored);
* two separate tables, one per event type (verbal 2-tuples, non-verbal 3-tuples), keyed by
  the **full** tuple (target included, `target_filtered` collapsed to a string when it has a
  single element);
* hypotheses 2–5 are the top-2 of each table (`collections.Counter.most_common`);
* **ties** would be broken by first-insertion order, which is deterministic here (sessions
  are visited in sorted-id order, events in file order). No tie occurs at either selection
  boundary: the non-verbal cut is 917 vs. 403 for the next candidate, and the verbal cut is
  86 vs. 82. The set is therefore stable, and it is what submission 825331 contains
  (verified: all 1302 cells of the submitted `anticipation.json` carry exactly this set).

## 2. Environment

Reproducing the submission requires **no third-party package** — Python ≥ 3.8, standard
library only (`requirements.txt` documents this; verified with Python 3.11 and 3.13.2, same
numbers). No GPU, no network, no random seed; the submission zip builds in ~2 s and the
cross-validation runs in ~2 s on one CPU core.

`requirements-dev.txt` lists the extra packages (numpy, torch, transformers,
sentence-transformers, decord) needed **only** to re-run the rejected development
experiments in `code/dev/`.

## 3. Paths that must be configured

| variable | needed for | default | used by |
|---|---|---|---|
| `UDIVA_HHOI_ROOT` | everything | `<datasets>/UDIVA-HHOI` | `code/udiva_data.py` |
| `UDIVA_T3_OUT` | output dir of the built zip | `./out` | `code/pack.py`, `code/make_test_sub.py` |
| `UDIVA_T3_WORK` | development experiments only (feature cache) | `<scratch>/t3` | `code/dev/*` |

`UDIVA_HHOI_ROOT` must point at the **decrypted dataset root**, i.e. the directory that
contains `development/` and `evaluation/`. Exactly three inputs are read:

* `development/annotated_sessions/annotations/<sid>.json` — the 21 annotated sessions
  (event frequencies + all ground truth);
* `development/annotated_sessions/starting_kit/anticipation/reference.json` — used to verify
  the ground-truth parser and to take the official evaluation grid of sessions
  `001080` / `181182`;
* `evaluation/eval_data/anticipation_template.json` — the released test query, filled in
  place to produce the submission.

The development-only scripts additionally read
`development/annotated_sessions/audiovisual/exo/GF/<sid>.mp4` and
`development/annotated_sessions/transcripts/<sid>.srt`.

## 4. Reproduce the final submission and the reported validation result

```bash
export UDIVA_HHOI_ROOT=/path/to/UDIVA-HHOI
cd code

# (a) the submitted zip: fills the released template, 651 segments x 2 participants
python make_test_sub.py b2          # -> out/test_b2_template.zip

# (b) the reported leave-one-session-out result of the SUBMITTED predictor
python cv.py

# (c) sensitivity of that result to the grid construction (see Section 5)
python cv.py protocols

# (d) ground-truth parser check against the official starting kit, and metric self-test
python udiva_data.py                # reproduces reference.json for 001080 / 181182
python sdl.py                       # scorer self-tests
```

(a) is fully deterministic and reproduces the payload of Codabench submission **825331**
**byte-identically** (387 381 bytes, sha256 `4f60ae91496f3d54…`).

(b) prints the reported result — the LOSO scores of the submitted predictor:

```
PROTOCOL: leave-one-session-out over 21 annotated sessions
  segments: 2835 (127 from the 2 official starting-kit grids, 2708 generated stride-2s event-containing windows)
  participant cells: 5670;  scorer: sdl.py (4 subtasks, best-of-K=5)

                                                 next verbal  nonvb   full  mean4   fstd  nseg
empty prediction (floor)                       0.0628 0.4975 0.1093 0.0628 0.1831 0.0266  2835
B2 = SUBMITTED predictor (825331)              0.3902 0.5760 0.3806 0.3041 0.4127 0.0392  2835
B4 = non-verbal-heavy variant (837289)         0.4114 0.4975 0.4261 0.3185 0.4134 0.0406  2835
```

`fstd` is the fold-to-fold standard deviation of `mean4` — the noise floor of this protocol
(≈ 0.039). Differences smaller than that are not resolvable with 21 sessions.

> Note on an earlier number: the value **0.4156** quoted in the first version of the fact
> sheet and of this README was the same predictor measured on the stride-4 s grid variant
> (`python cv.py protocols`, row 3), not on the protocol shipped as the `cv.py` default.
> The reported result is now the default protocol throughout: **mean4 = 0.4127**
> (next 0.3902 / verbal 0.5760 / non-verbal 0.3806 / full 0.3041).

## 5. Validation protocol

* **Folds.** 21 leave-one-session-out folds over the 21 annotated development sessions
  (the only sessions with ground truth). Every predictor is refit on the 20 training
  sessions of a fold; scored segments come only from the held-out session. Fold predictions
  are pooled over all 2835 segments for the reported numbers.
* **Evaluation timestamps.** The official starting kit ships an anticipation grid for only
  2 of the 21 sessions (`001080`: 60 segments, `181182`: 67 segments, both at ~4 s spacing).
  Those two grids are used **verbatim** (127 segments). For the remaining 19 sessions a grid
  is generated (`udiva_data.make_grid`): non-overlapping 2 s windows
  `t_b = 4.0, 6.0, 8.0, …`, `t_e = t_b + 2 s`, up to the end of the last annotated event,
  keeping the windows that contain **at least one annotated event of at least one
  participant** (2708 segments). Total 2835 segments = 5670 (segment, participant) cells.
* **Ground truth.** Parsed from the raw annotations with the rule verified against the
  official reference: an event belongs to the window iff `t_b < start <= t_e`; events are
  ordered by `(start, end)` within a participant; verbal → `[utterance_type, target]`,
  non-verbal → `[high_level_action, low_level_action, target]`.
* **Scorer.** `code/sdl.py`, our re-implementation of the published normalized-SDL
  definition (substitution weights 0.8/0.2 verbal, 0.4/0.4/0.2 non-verbal, insertion =
  deletion = 1, transposition = 0.5, `score = 1 − D/max(m,n)`, best-of-K per participant and
  **per subtask**, mean over participants then over segments). `mean4` = unweighted mean of
  the four subtask columns. `python sdl.py` runs its unit tests.

### Why event-free windows are dropped, and how that differs from the challenge grid

The filter keeps the generated grid comparable to the two official grids and to the released
test query, which are **not** uniform sweeps of the session: in both official dev grids
**0 % of the segments are event-free** (event-free *participant cells* do occur: 8.3 % and
3.7 %), and the released test template starts at `t_b = 2.0` with a median spacing of 2.66 s
— i.e. the challenge grid is itself concentrated on annotated activity. A uniform 2 s sweep
of a session would additionally contain windows in which nothing at all is annotated, where
the empty hypothesis scores exactly 1.0 for both participants and every method looks better.

The effect is small and measurable, because annotation coverage is dense: the filter removes
only **1.0 %** of candidate windows (28 of 2736 at stride 2 s), and including them raises the
submitted predictor from `mean4` 0.4127 to 0.4184 (`python cv.py protocols`) — i.e. ≈ +0.006,
one seventh of the fold-to-fold noise. The reported (filtered) protocol is thus the slightly
**conservative** choice, and no conclusion in Section 6 changes if the filter is switched
off. Grid sensitivity overall (all rows from `python cv.py protocols`):

| grid | next | verbal | non-verbal | full | mean4 | segments |
|---|---|---|---|---|---|---|
| **reported**: official(2) + stride 2 s, event-containing | 0.3902 | 0.5760 | 0.3806 | 0.3041 | **0.4127** | 2835 |
| stride 3 s, event-containing | 0.3858 | 0.5723 | 0.3853 | 0.3031 | 0.4116 | 1933 |
| stride 4 s, event-containing (official density) | 0.3975 | 0.5752 | 0.3853 | 0.3054 | 0.4159 | 1488 |
| stride 2 s, all windows (no filter) | 0.3961 | 0.5801 | 0.3866 | 0.3109 | 0.4184 | 2863 |
| stride 4 s, all windows (no filter) | 0.4023 | 0.5786 | 0.3903 | 0.3110 | 0.4205 | 1500 |

Local↔server gap: the server scored this predictor at mean-of-4 = 0.447 versus 0.4127 here
(+0.034, within the fold noise); a constant predictor cannot overfit a grid, so the gap is
attributed to the composition of the hidden test sessions.

## 6. Development experiments (NOT part of the submission)

`code/dev/` contains every input-conditioned approach that was evaluated during development
and rejected, plus the alternative-set variants. **None of them is part of the final
CodaBench submission**, which is the constant predictor of Section 1.

`code/dev/RESULTS.md` holds the complete comparative table — the four subtask scores and
their mean for every approach, all recomputed under the single protocol of Section 5 — plus
the exact pretrained models, observation windows, sampling, prediction rules and
hyperparameters. The conclusion in one line: no implementable conditioned variant beats the
constant prior, and even the oracle variants (which read the ground-truth past) stay within
the 0.039 fold-noise of it.

```bash
cd code
python dev/altsets.py        # K=5 set variants: longer / mixed alternatives    (~10 s)
python dev/priors.py         # role / time-bin / per-session priors             (~5 s)
python dev/cond.py           # event-history Markov + ongoing-activity oracles  (~20 s)
python dev/tune.py           # greedy K=5 composition search (provenance of B2)
python dev/tune3.py          # exhaustive K=5 search over a curated pool
python dev/explore.py        # dataset landscape (event/length/empty-rate statistics)
# representation probes: need requirements-dev.txt (+ a GPU for the video extraction)
python dev/feat_extract.py all   # DINOv2 features of the exo GF video -> $UDIVA_T3_WORK/feats
python dev/txt_extract.py        # transcript sentence embeddings      -> $UDIVA_T3_WORK/txtfeats
python dev/vidprobe.py video ; python dev/vidtrain.py video
python dev/vidprobe.py text  ; python dev/vidtrain.py text
```

## 7. File map

| file | role |
|---|---|
| `code/udiva_data.py` | annotation parsing, event tuples, grid construction, **all configurable paths** |
| `code/sdl.py` | normalized-SDL scorer + 4 subtask reductions (self-tested) |
| `code/cv.py` | LOSO CV harness; `python cv.py` = reported result, `python cv.py protocols` = grid sensitivity |
| `code/predictors.py` | the predictors; `b2_factory` = **submitted**, `b4_factory` = A/B variant |
| `code/pack.py`, `code/make_test_sub.py` | fill the released template, write the submission zip |
| `code/dev/` | development-only experiments (rejected families) + `RESULTS.md` |
| `requirements.txt`, `requirements-dev.txt` | submission (stdlib only) / development dependencies |
| `requirements_shared_venv.txt` | reference only: full frozen list of the cluster venv used during the challenge |

## 8. License

MIT (see `LICENSE`); the code remains publicly accessible for at least three years.
