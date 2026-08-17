# UDIVA-HHOI Track 1 — Multimodal Exocentric Event Recognition

Team **JLShen** (Codabench user `junlong`) · Competition 15954 · Test phase 26330
Final submission: **821839** — server mAP **0.0318** (verbal 0.0237 / non-verbal 0.0398).

Complete training + inference code to reproduce submission 821839 from scratch
(raw dataset → features → trained models → `recognition.json` zip).

## Method

Per 2-second segment, recognition is cast as candidate generation + metric-aware reranking.
Candidates are the event tuples observed in the training annotations (`min_count>=2`), scored
by channel-specific base models — verbal: multilingual-e5-base embeddings of the segment
transcript + logistic heads; non-verbal: an 11-seed MLP ensemble over frozen VideoMAE-large
clip features — blended with training priors (position-conditioned priors enter the
reranker as a feature). A HistGradientBoosting
reranker per channel (verbal: segment features + a video-derived target feature + 13
hand-crafted audio features; non-verbal: cross-segment context features) rescores all
candidates; final confidence = 3-seed reranker average. Decoding emits all candidates with
their confidences (mAP is threshold-free). Trained on the 21 annotated development sessions
only.

## Environment

- Python 3.11.5; packages pinned in `requirements_shared_venv.txt`
  (torch 2.5.1, transformers 4.49, decord, librosa, scikit-learn, sentence-transformers).
- One >=16GB GPU for feature extraction; CPU suffices for training the heads/rerankers.
- Pretrained models, downloaded by name from HuggingFace and run locally:
  `MCG-NJU/videomae-large-finetuned-kinetics` (1024-d), `intfloat/multilingual-e5-base`.

## Reproduce submission 821839

Data: `DEV` = `development/annotated_sessions` (21 sessions), `EVAL` = `evaluation/`
(7 sessions). Adjust the path constants at the top of the scripts.

```bash
EVAL_SIDS=005013,020025,027113,035040,041083,044156,066067
VM=MCG-NJU/videomae-large-finetuned-kinetics   # 1024-d; the shipped backbone (default in extract_feats.py)

# 1. VideoMAE features — dev (21 sessions), then the 7 eval sessions (eval 2s grid = video duration)
python code/extract_feats.py --sessions all       --model $VM --out <DEV_FEAT>
python code/extract_feats.py --sessions $EVAL_SIDS --model $VM --eval \
       --gf_dir <EVAL>/audiovisual/exo/GF --out <EVAL_FEAT>
# 2. audio features — dev, then eval
python code/extract_audio.py --sessions all       --out <DEV_AUDIO>
python code/extract_audio.py --sessions $EVAL_SIDS --eval \
       --gf_dir <EVAL>/audiovisual/exo/GF --out <EVAL_AUDIO>
# 3. train on the 21 dev sessions, predict the 7 eval sessions, pack the zip
#    (point DEV_FEAT/EVAL_FEAT/DEV_AUDIO/EVAL_AUDIO at the top of ship_eval.py to the paths above)
python code/ship_eval.py    # -> recognition.json at the zip root == submission 821839
```

All seeds are fixed in code (11-seed base ensemble; reranker seeds {0,1,2}); feature
extraction is deterministic. Local validation: leave-one-session-out CV over the 21 annotated
sessions with our re-implementation of the official mAP (pooled CV 0.0229).

## Pipeline

`code/ship_eval.py` is the single end-to-end entry point; the stages run **sequentially**:

```
transcript ─▶ e5 encoder ─▶ per-attr LR heads ─┐
                                                ├▶ ×prior blend ─▶ HGB reranker ─▶ emit all ─▶ recognition.json
GF clip ─▶ 3 crops ─▶ VideoMAE-L ─▶ 11-seed MLP ┘   (+audio, video→target, x-seg feats; 3-seed mean)
```

| stage | function → module |
|---|---|
| GT parse + 2 s grid | `udiva.build_all_gt`, `seg_grid` |
| video features (3 crops) | `extract_feats.py` |
| audio features (13) | `extract_audio.py` |
| verbal base (e5 + per-attr LR heads) | `verbal_emb.embed_all/fit_heads/proba_rows` |
| verbal candidates + static prior | `verbal_v2.build_cands` |
| non-verbal base (11-seed per-attr MLP) | `nv_mlp2.{build_candidates,train_fold}`, `nonverbal_model.instances` |
| session-position prior | `integ_tp.taus_of/time_table` |
| verbal reranker rows | `rerank_v7.dump_verbal_v7` → `rerank_v8.add_audio` |
| non-verbal reranker rows | `rerank_v3.dump_nonverbal` + `rank_exp.xseg_feats` |
| reranker + decode | `ship2.train_hgb/score_hgb/decode_emit` |
| pack zip | `pack.to_submission/write_zip` |

**Complementary, not alternative.** The three scripts previously grouped under "reranker
features" apply *in order* to build one feature matrix: `rerank_v7` builds the verbal rows (base
features + a video→target feature), `rerank_v8` **appends** the 13 audio features, and `rank_exp`
**concatenates** the non-verbal cross-segment features. They all feed the single reranker.

**Reranker.** One *pointwise* scikit-learn `HistGradientBoosting` classifier per channel; the
training target is `y = 1` iff the candidate tuple exactly matches a ground-truth event in the
segment (all attributes), else `0`; the non-score features are inputs only. The final confidence
is the arithmetic mean of 3 seeds' `predict_proba`. (Not RankNet — a pairwise variant was scouted
and dropped.)

`rerank.py` / `rerank_v3` / `rerank_v4` / `rerank_v7` / `rerank_v8` are an incremental
**experiment lineage**: each file's `main()` is a leave-one-session-out CV ablation, and the
shipped path imports only the specific feature-builder functions in the table above. Every file
carries a role header stating its place in the shipped pipeline.

Submission 821839 on Codabench is the exact submitted artifact for comparison.

## License

MIT (see `LICENSE`); the code remains publicly accessible for at least three years.
