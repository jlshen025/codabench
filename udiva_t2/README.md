# UDIVA-HHOI Track 2 — Multimodal Egocentric Event Recognition

Team **JLShen** (Codabench user `junlong`) · Competition 16643 · Test phase 27267
Ranked submission **829988** — server mAP **0.0178** (verbal 0.0200 / non-verbal 0.0155),
cross-validated mAP ≈ 0.0137 on the 21 annotated development sessions.

Complete training and inference code for that entry. Everything is trained on the 21 annotated
development sessions only; no external data, no unannotated sessions, no pseudo-labels.

## 1. Method

Two **independent** channels per predefined 2-second segment. `docs/pipeline.pdf` is the same
diagram as Fig. 1 of the fact sheet; the ASCII sketch below is its abbreviated form.

```
NON-VERBAL  (E1 + E2 egocentric video)                VERBAL  (transcript .srt)
  16 frames per 2 s segment, per view                   cue text  t = "[spk] <prev cue> </s> <cue>"
      |                                                     |
  +-- VideoMAE-L / K400        frozen  --+              TF-IDF: word (1,2) + char_wb (2,5)
  +-- VideoMAE-B / SSv2        frozen  --+                  |                    |
  +-- TimeSformer-HR / SSv2    frozen  --+            34 logistic heads   108 logistic heads
  |     x = [f_E1 ; f_E2]  (concat, L2)                 -> P(u|t)          -> P(tau|t)
  |         |                                               \                  /
  |   64 one-vs-rest logistic heads, one per                 \                /   rho = speaker
  |   JOINT class (rho, h) -> P_b(rho,h | x)                  \              /         tag
  |         |                                        conf = P(u|t) * P(tau|t) * (0.5 + P(tau|u))
  +-- VideoMAE-L / K400, last 2 blocks fine-tuned     m = argmax P(m | u,tau);  gate P(tau|u) > 0
  |     x = 0.5 * (f_E1 + f_E2)  (average, in-net)               |
  |     -> head -> sigmoid -> P_4(rho,h | x)          copy each cue event into EVERY 2 s
  |         |                                         segment the cue overlaps (max per tuple)
  |   S(rho,h) = SUM of the 4 posteriors, equal weights          |
  |         |   (fusion happens here, BEFORE tuple composition)  |
  |   top-14 (rho,h) with S >= 0.02, x 3 low-level x 4 target    |
  |   conf = S * (0.5 + P(l|rho,h)) * (0.5 + P(tau|rho,h))       |
  |   m = argmax P(m | rho,h)                                    |
  v                                                              v
  non-verbal events (rho, h, l, tau, m) + confidence    verbal events (rho, u, tau, m) + confidence
                    \                                          /
                     -> one JSON: {"verbal": {...}, "nonverbal": {...}}, scored separately
```

Symbols: `rho` subject · `h` high-level action · `l` low-level action · `tau` target ·
`m` modifier · `u` utterance type. All priors `P(.|.)` are training co-occurrence frequencies.
Note that the non-verbal heads are defined over the **joint** class `(rho, h)` — there is no
separate subject head and no separate action head — and that the target prior is conditioned on
`(rho, h)`, not on the low-level action.

## 2. Where each stage lives

| pipeline stage | code |
|---|---|
| 2 s segment grid, ground truth, transcripts, video paths | `code/udiva/parse.py`, `code/udiva/data.py` |
| 16 frames per segment/view → frozen clip features | `code/extract_feats.py` |
| view fusion by concatenation (frozen backbones) | `models_video.load_feats` |
| 64 one-vs-rest heads over `(rho, h)` | `models_video.fit_pair_heads`, `.predict_pair_probs` |
| fine-tuned 4th component (view fusion by averaging) | `code/ft_video.py` (train), `code/ft_infer.py` (test-time posteriors) |
| equal-weight late fusion of the 4 posteriors | `models_video.fuse_pair_probs`, `.fuse_ft_probs` |
| non-verbal tuple composition with priors | `models_video.compose_nonverbal` |
| cue text, TF-IDF, utterance-type + target heads | `models_text.VerbalCueModel.fit` |
| verbal composition, co-occurrence gate, cue → segment | `models_text.VerbalCueModel.predict` |
| submission JSON and zip | `code/predict_submission.py` |
| official mAP re-implementation, leave-sessions-out CV | `code/udiva/metric.py`, `code/udiva/cv.py` |
| CV of the submitted configuration | `code/run_ensemble_cv.py` |

(`models_video` = `code/udiva/models_video.py`, `models_text` = `code/udiva/models_text.py`.)

## 3. Reproducing submission 829988

Data paths are constants at the top of `code/udiva/data.py` (`ROOT` for
`development/annotated_sessions`, `EVAL_ROOT` for `evaluation/`) — edit them first. Below,
`$T2` is any working directory for features and checkpoints, and `$EVAL` is the evaluation
split.

```bash
# 1. frozen clip features: 3 backbones x {21 annotated, 7 evaluation} sessions x {E1, E2}
#    (--batch 4 for TimeSformer-HR, whose input is 448 px)
python code/extract_feats.py --sids all  --split annotated \
    --model MCG-NJU/videomae-large-finetuned-kinetics --out $T2/feats/videomae_large       --batch 16
python code/extract_feats.py --sids all  --split annotated \
    --model MCG-NJU/videomae-base-finetuned-ssv2      --out $T2/feats/videomae_ssv2        --batch 16
python code/extract_feats.py --sids all  --split annotated \
    --model facebook/timesformer-hr-finetuned-ssv2    --out $T2/feats/timesformer_ssv2     --batch 4
python code/extract_feats.py --sids eval --split eval \
    --model MCG-NJU/videomae-large-finetuned-kinetics --out $T2/feats_eval/videomae_large  --batch 16
python code/extract_feats.py --sids eval --split eval \
    --model MCG-NJU/videomae-base-finetuned-ssv2      --out $T2/feats_eval/videomae_ssv2   --batch 16
python code/extract_feats.py --sids eval --split eval \
    --model facebook/timesformer-hr-finetuned-ssv2    --out $T2/feats_eval/timesformer_ssv2 --batch 4

# 2. the fine-tuned fourth component: train on all 21 sessions, then run it on the eval videos
export FT_MODEL=MCG-NJU/videomae-large-finetuned-kinetics
python code/ft_video.py --train_all --epochs 11 --batch 4 --n_unfreeze 2 \
    --save_model $T2/ftL_all21.pt
python code/ft_infer.py --model_pt $T2/ftL_all21.pt --sids eval --split eval --batch 8 \
    --out $T2/feats_eval/ftL_eval.npz

# 3. build the submission zip  (this exact command produced submission 829988)
python code/predict_submission.py \
    --feat_dir $T2/feats_eval/videomae_large \
    --ens_feats "videomae_large:1,videomae_ssv2:1,timesformer_ssv2:1" \
    --dev_feat_root $T2/feats --test_feat_root $T2/feats_eval \
    --ft_oof $T2/feats_eval/ftL_eval.npz \
    --trans_dir $EVAL/transcripts \
    --out submission/t2_ftens.zip
```

The zip holds four identical copies of the predictions JSON
(`predictions.json`, `recognition.json`, `answer.json`, `reference.json`): the file name the
scorer expects was not documented and both phases hid their output, so we hedged. Every event
carries the same value under `score` and `confidence`, for the same reason. If the organisers'
segment template is available, pass it as `--grid_json <template>` and its exact session and
segment keys are used instead of the 2-second duration grid.

Runtime on one NVIDIA H100 MIG slice (20 GB): ~3–5 h for step 1 (28 sessions × 3 backbones),
~2 h for step 2, minutes for step 3. Steps 1–2 need a GPU; step 3 is CPU-only.

## 4. Local validation

Model selection used leave-sessions-out CV over the 21 annotated sessions, pooling all
out-of-fold predictions and scoring them once with our re-implementation of the official mAP
(`code/udiva/metric.py`, validated against `reference.json`: a perfect prediction scores 1.0,
a shuffled one ~0.002). Server scores were hidden in both challenge phases, so this was the only
selection instrument.

```bash
# CV of the submitted configuration (~0.0137). Needs the fine-tuned component's out-of-fold
# posteriors, i.e. one fine-tune per fold, with the SAME k and seed:
for i in 0 1 2 3 4 5 6; do
  python code/ft_video.py --fold $i --k 7 --epochs 12 --batch 4 --out_npz $T2/feats/ftL_fold$i.npz
done
python code/run_ensemble_cv.py --feat_root $T2/feats \
    --backbones videomae_large,videomae_ssv2,timesformer_ssv2 \
    --ft_glob "$T2/feats/ftL_fold*.npz" --k 7 --seed 0

# the fine-tuned component alone, without the frozen backbones
python code/run_ft_cv.py
```

Measured CV mAP: best single model 0.0112, frozen-only three-backbone ensemble 0.0127, and
0.0137–0.0138 with the fine-tuned component added — the fine-tune was worth only +0.0005 as a
*replacement* for the frozen large backbone, but +0.001 as an ensemble *partner*.

## 5. Environment

- Python 3.11.5; `requirements_shared_venv.txt` is the full pinned environment of our cluster
  (torch 2.5.1, transformers 4.49, scikit-learn 1.8, decord/PyAV). Only `numpy`, `scipy`,
  `scikit-learn`, `torch`, `transformers` and `decord` are actually imported.
- One GPU with ≥16 GB for feature extraction and the fine-tune; step 3 runs on CPU.
- Pretrained weights are downloaded from HuggingFace by name and run locally:
  `MCG-NJU/videomae-large-finetuned-kinetics`, `MCG-NJU/videomae-base-finetuned-ssv2`,
  `facebook/timesformer-hr-finetuned-ssv2`. No dataset content ever leaves the local machine.
- Seeds are fixed in code.

## 6. Contents

```
code/extract_feats.py        frozen clip features for one backbone (all three use this script)
code/ft_video.py             partial fine-tune of VideoMAE-large (the 4th component)
code/ft_infer.py             that checkpoint applied to the evaluation videos
code/predict_submission.py   builds the submitted zip
code/run_ensemble_cv.py      CV of the submitted configuration
code/run_ft_cv.py            CV of the fine-tuned component alone
code/verbal_ft.py            ABLATION, not part of the entry (see below)
code/udiva/parse.py          raw annotations -> the 2 s "recognition" reference format
code/udiva/data.py           sessions, ground truth, transcripts, video paths
code/udiva/metric.py         re-implementation of the official mAP
code/udiva/cv.py             leave-sessions-out folds and pooled out-of-fold scoring
code/udiva/models_video.py   non-verbal channel: features, heads, fusion, tuple composition
code/udiva/models_text.py    verbal channel: cue model, two head banks, tuple composition
docs/pipeline.{pdf,tex}      the pipeline figure
```

`code/verbal_ft.py` fine-tunes `xlm-roberta-base` on utterance-type recognition. It improved
utterance-type macro-AP from 0.163 to 0.22 in cross-validation, but it is **not** part of the
submitted entry: the verbal channel of submission 829988 is the TF-IDF cue model in
`udiva/models_text.py`. It is kept because the fact sheet reports the ablation.

Verified after cleaning this archive: re-running its verbal channel on the evaluation
transcripts reproduces the 48 847 verbal events of submission 829988 exactly — identical tuple
sets and identical confidences.

## 7. License

MIT (see `LICENSE`). The code remains publicly accessible for at least three years.
