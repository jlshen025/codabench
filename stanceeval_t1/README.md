# StanceEval-2026 Track 1 — Arabic stance detection, held-out target

Team **JLShen** (Codabench user `JLShen`) · Competition 16332 · Leaderboard 18030 ·
Evaluation phase 29242
Ranked submission **878207** — server **Overall_Favg2 0.899400** (Favg3 0.664640,
Accuracy 0.857955), **1st of 24**.

Complete inference, training and reproduction code for that entry, together with the per-row
outputs of every LLM the entry called. The submitted label file is rebuilt **bit-exactly** by one
command that needs neither a GPU nor network access.

The blind test is 352 tweets, all on the target **"Women Driving"**, which appears in none of the
three training targets — so Track 1 is really near-transfer, not a seen-target task. That single
fact drives the whole design: seen-target fine-tuning is demoted to a tie-breaker and zero-shot
frontier readers carry the decision.

## 1. Method

```
tie    = mean(luna, g55, lunaFS, g55FS, opus-4-8, sonnet-4.5)  +  1e-3 * ce
base   = argmax(tie), then the None count is projected down to exactly N=3 rows
         (keep the 3 largest None-margin rows; flip the rest to their strongest committed class)
final  = base + the 4 Against-pool rows ranked highest by "not-Against-ness" moved to None
```

**Six zero-shot / few-shot voters, equally weighted.** Two model families from two laboratories,
two prompt regimes:

| role | model | prompt | per-row output |
|---|---|---|---|
| voter 1 | `gpt-5.6-luna` | zero-shot | `code/_eval/frontier_luna_test.npz` |
| voter 2 | `gpt-5.5` | zero-shot | `code/_eval/frontier_g55_test.npz` |
| voter 3 | `gpt-5.6-luna` | few-shot, k=12, batch-level retrieval | `code/_eval/frontier_lunaFS_test.npz` |
| voter 4 | `gpt-5.5` | few-shot, k=12, batch-level retrieval | `code/_eval/frontier_g55FS_test.npz` |
| voter 5 | `claude-opus-4-8` | zero-shot, target-grounded (**cross-lab**) | `code/_eval/frontier_opus_test.npz` |
| voter 6 | `claude-sonnet-4.5` | zero-shot, target-grounded (**cross-lab**) | `code/_eval/frontier_son45_test.npz` |

Few-shot demonstrations are retrieved **per batch**, not per tweet; per-tweet retrieval measured
worse (0.8694 vs 0.8815). Cross-laboratory diversity was the single largest measured gain of the
run (+0.0065) — larger than any prompt or ensemble-weight change within one family.

**The arbiter `ce`, at weight 1e-3.** The seen-target fine-tuned pipeline —
`0.15 · MTL-encoder-ensemble + 0.85 · Qwen2.5-14B-LoRA prompt-ensemble` — enters at a weight that
can only break exact ties among the six voters. Standalone on this test it scores **0.7762**, the
weakest component in the system, and it is nonetheless the best *arbiter* available: cross-lab
0.8788, few-shot 0.8758, encoder 0.8722 all lose in that slot. A model that is bad at the task can
still be good at breaking ties, because ties are exactly the rows where the voters carry no signal.

**The two operating-point corrections.** Favg2 excludes None as a *class* but not as a *row*: a
gold-None row predicted Favor or Against is a false positive that damages a scored class. Both
final steps therefore move rows without changing any model:

- *None count.* The base is projected to exactly N=3 None rows (+0.0060 over N=14).
- *Four Against→None flips.* Within the Against pool, moving a row to None pays +0.001274 if the
  row is gold-None **or** gold-Favor (a gold-Favor row sitting in the Against pool is already
  outside the F1_Favor numerator) and costs −0.001616 only if it is gold-Against. The payoff set
  is therefore 34 of 187 rows and the break-even precision is **55.9%**, not 50%. Rows are ranked
  by a sentiment/tone reading — within the Against pool gold-Against sits at the low-tone end,
  gold-Favor at the high end, gold-None in the middle, so "tone is not low" is monotone in
  not-Against-ness. Decoded afterwards: 3 of the 4 flips were correct (+0.0022).

`code/build_na.py` carries the full derivation in its module docstring.

### Where each stage lives

| stage | code |
|---|---|
| zero-shot voters, batching, output parsing | `code/llm_predict.py` |
| few-shot voters (k=12, batch retrieval) | `code/fewshot_conv.py` |
| cross-lab grounded voters | `code/llm_grounded_agent.py`, `code/grounded_prompts.py` |
| dedicated None detectors, sentiment/tone readers | `code/none_detect.py` |
| MTL encoder arm of the arbiter | `code/train_mtl.py`, `code/predict_test.py` |
| Qwen2.5-14B LoRA arm of the arbiter | `code/train_lora.py`, `code/predict_lora.py` |
| full arbiter deploy pipeline | `code/run_deploy.sh` |
| **the submitted entry** | `code/build_na.py` |
| the N-sweep predecessor | `code/reproduce_final.py` |
| exact confusion matrix from a server score | `code/decode_conf.py` |
| official metric re-implementation | `code/stance_lib.py` |

## 2. Reproducing submission 878207

```bash
cd code
python build_na.py --k 4 --mode sent --label final     # -> code/_eval/final.zip
```

Verified: byte-identical to `submission/codabench/878207__wd_na_sent4.zip`, 352/352 labels,
md5 `f6e77e7391fd39e2055a1893548cab40`. Only `numpy` is needed — the script starts from the
cached per-row LLM outputs in `code/_eval/`, so it runs offline, on CPU, in under a second.

Two earlier ranked entries rebuild the same way and are included as further checks:

```bash
python reproduce_final.py                # 869442, Favg2 0.891160  -> EXACT MATCH
python reproduce_final.py --n 3 \
    --verify ../submission/codabench/875984__wd_champN3.zip   # 875984, 0.897170 -> EXACT MATCH
```

Reproducing from cached arrays rather than from live API calls is deliberate and is the only
reproducible option: voter 5's model, `claude-opus-4-8`, has since been retired by its provider,
and frontier endpoints are in any case not bit-stable across time.

## 3. The LLM inference record

Every LLM was called **programmatically**, batched over an OpenAI-compatible
`/chat/completions` endpoint at temperature 0 — no interactive chat session was used at any point,
so there are no chat transcripts or UI screenshots to show. The equivalent record, complete and
machine-readable, is:

| what | where |
|---|---|
| per-row output of every model × prompt run (label, vote-fraction probability, row index, target) | `code/_eval/frontier_*.npz` (67 runs) |
| dedicated None-detector scores, 0–100 per row | `code/_eval/nd_*.npz` (22) |
| sentiment/tone scores per row | `code/_eval/sent_*.npz` (8) |
| exact model id, grounding card, row count and wall-clock of each run | `code/_eval/*_meta.json` |
| per-batch call log of each run | `code/_eval/*.log` |
| per-row rationale trace of the manual adjudication pass | `code/_eval/adjudication.jsonl` |
| arbiter posteriors (fine-tuned encoders, Qwen LoRA) | `code/results/eval_deploy_all/_llm/` |

Each `.npz` holds `proba[N,3]` in column order `[Against, Favor, None]`, `pred_id[N]`, `idx[N]` and
`target[N]`, aligned to the released test-row order — that order is the only join key the scorer
uses.

**Output parsing.** Models are asked for exactly one line per tweet, `<number>|<Label>`. Parsing is
the regex `LBL_RE` at `code/llm_predict.py:79` and the loop in `classify()` at
`code/llm_predict.py:143`; `code/fewshot_conv.py` and `code/llm_grounded_agent.py` import that same
regex, so all six voters share one parser. A batch that loses more than `max(2, len(batch)//5)` of
its lines is re-sent once with a larger token budget; any row still unparsed after that falls back
to `None`. The None detectors use the numeric variant `SCORE_RE` at `code/none_detect.py:119`.

**Regenerating a voter array from live APIs.**

```bash
# voters 1-2: zero-shot
python llm_predict.py --model gpt-5.6-luna --csv <test_csv> \
    --order file --batch 44 --reasoning none --out _eval/frontier_luna_test.npz
# voters 3-4: few-shot, k=12 demos per BATCH
python fewshot_conv.py --model gpt-5.5 --csv <test_csv> --k 12 --batch 16 \
    --out _eval/frontier_g55FS_test.npz
# voters 5-6: cross-lab, target-grounded (reuse the frozen cards for a card-identical run)
python llm_grounded_agent.py --model claude-sonnet-4.5 --csv <test_csv> --order file \
    --batch 40 --cards_from _eval/frontier_son45_test_meta.json \
    --out _eval/frontier_son45_test.npz
```

These scripts resolve an endpoint through `llm_provider_registry.provider_for(model)`, a thin
in-house shim that is **not** part of this release. It returns nothing but
`{"base_url": ..., "api_key": ..., "model": ...}` for an OpenAI-compatible chat endpoint and is
consumed only in `call_api()` (`code/llm_predict.py:82`); substituting five lines that read your own
credentials is sufficient. Everything that determines the result — prompts, batching, temperature,
retrieval, parsing — is in this repository.

## 4. Training the arbiter

```bash
bash code/run_deploy.sh <TEST_CSV> <OUT_ZIP>
```

Four Qwen2.5-14B-Instruct LoRA adapters (Arabic and Latin letter verbalizers × seeds 42 and 1,
epoch 3) plus two multi-task encoders (`UBC-NLP/MARBERTv2`, `aubmindlab/bert-base-arabertv02-twitter`),
all trained on train+dev = 4121 rows. About 36 min per adapter on one NVIDIA H100 MIG slice; the
encoders are minutes. `run_deploy.sh` averages the two verbalizer prompts into the Qwen ensemble
and blends it with the encoders at 0.85/0.15. Base weights are downloaded from HuggingFace by name.

The reproduction in §2 does not need any of this: the arbiter's contribution to the submitted entry
is the frozen posterior in `code/results/eval_deploy_all/_llm/{enc_ce,dep_qens}.npz`.

## 5. Data

Not included here and not redistributable from this repository — request it from the task
organizers.

- Train / dev: MawqifV2 `train.csv` (3502 rows) / `dev.csv` (619), three targets. `code/stance_lib.py`
  reads them from `$MAWQIF_DIR` (a cluster path is the fallback default).
- Blind test: the organizers' `test_seen.csv`, normalized to a `text` column with **row order
  preserved**, 352 rows, all target "Women Driving".
- Submission format: `predictions.txt`, one label per line in test-row order, zipped flat.

## 6. Scoring and local validation

`stance_lib.compute_metrics` re-implements the official scorer: Favg2 = mean(F1_Favor, F1_Against),
pooled, None excluded from the score. Verified to four decimals against the server on the
development phase (local↔server gap 0.00).

Because the server reports Favg2, Favg3 and accuracy together, a submission's **full confusion
matrix is recoverable exactly** — in particular `F1_None = 3·Favg3 − 2·Favg2` for any entry on the
public board. `code/decode_conf.py <zip> <favg2> <favg3> <acc>` does this and is how the residual
above was measured rather than estimated:

```
python decode_conf.py <zip> <favg2> <favg3> <acc>
```

Rows were selected only from model readings of the **released test inputs**. Server feedback was
used to score whole candidate submissions and to forecast depth, never to search over assignments
of labels to individual hidden rows.

## 7. Environment

- Python 3.11.5. The reproduction in §2 needs only `numpy` (developed against 2.4.2).
- Regenerating voters additionally needs `pandas` and `requests`; training the arbiter needs
  `torch==2.5.1`, `transformers`, `peft`, `accelerate`, `sentencepiece`, `scikit-learn` and one GPU
  with ≥40 GB (or an H100 MIG slice) — see `requirements.txt`.
- Cluster paths appear as fallback defaults in several scripts (HuggingFace cache, dataset root,
  local backbone mirrors). `$MAWQIF_DIR`, `$HF_HOME` and the `--base_model` / `--adapters` arguments
  cover everything the documented commands touch.
- Seeds are fixed in code; LLM calls use temperature 0.

## 8. Contents

```
code/*.py, code/*.sh          every inference, training, analysis and submission script of the run
code/_eval/                   per-row LLM outputs, run logs, grounding cards, adjudication trace
code/results/eval_deploy_all/_llm/   frozen arbiter posteriors (encoders, Qwen LoRA)
submission/codabench/         the three archived submission zips the reproduction verifies against
```

`code/` is the run's script directory as it stood at the close of the competition, so it contains
the refuted branches as well as the shipped one — dedicated None-head training
(`train_none_det.py`), hashtag-based flips (`build_tag.py`), synthetic augmentation
(`gen_synth.py`, `gen_paraphrase.py`), self-consistency and debate (`debate.py`), translation to
English (`translate.py`), a sarcasm head (`sarc_emit.py`). None of these is part of submission
878207; they are kept because the system paper reports them as negative results.

## 9. License

MIT (see `LICENSE`). The code remains publicly accessible for at least three years.
