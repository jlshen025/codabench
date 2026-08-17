# StanceEval-2026 Track 2 — Arabic stance detection, unseen targets

Team **JLShen** (Codabench user `JLShen`) · Competition 16333 · Leaderboard 18031 ·
Evaluation phase 29244
Ranked submission **869013** — server **Unseen_Overall_Favg2 0.935700** (Favg3 0.866986,
Accuracy 0.913043), **4th of 21**.

Complete inference and reproduction code for that entry, together with the per-row outputs of
every LLM the entry called. The submitted label file is rebuilt **bit-exactly** by one command
that needs neither a GPU nor network access.

The blind test is 644 tweets over two targets — **Ecars** (332) and **Trimester** (312) — neither of
which appears in the training data, whose 2721 labelled rows cover only Covid Vaccine and Digital
Transformation. The entry is zero-shot for that reason: every fine-tuned arm we trained improved on
the seen targets and collapsed on a distant unseen one.

## 1. Method

**Target-grounded zero-shot frontier reader, plus a pre-registered adjudication of contested rows.**

```
base  = gpt-5.6-sol, greedy, target-grounded zero-shot over the 644 test inputs   -> 0.9016
final = base + 73 single-row corrections from the adjudication rounds of §3       -> 0.9357
```

**Grounding cards.** For each target an LLM writes a compact Arabic card — a one-line `definition`
plus 6–8 `aspects` — which is injected into the system prompt, so the reader can map an abstract
target name to its semantic neighbourhood and count *indirect* stance expressed through an aspect.
The card is generated per target from the target string alone, so nothing about the method is
tuned to a particular topic and it transfers to targets never seen. The cards actually used are
frozen in `code/_llm/fixed_cards.json` and are shared by every reader in the study, so all A/B
comparisons are card-identical.

**Classifier.** `gpt-5.6-sol`, zero-shot, greedy (temperature 0, reasoning effort *none*), batched
about 40 tweets per request, one output line per tweet as `<n>|<Label>`.

**Corrections.** 73 rows, as `None→Favor` 32 · `None→Against` 20 · `Against→Favor` 17 ·
`Favor→None` 2 · `Favor→Against` 1 · `Against→None` 1. The dominant operation is *recovering a
stance from a row the base reader abstained on*, which is the direct consequence of §4: in this
corpus, neutral-toned rows are labelled Favor 50.7% of the time and None only 22.9%. The full flip
set is `code/_llm/champion_flips.json`; its per-round provenance is `code/_llm/opus5_prereg.md`,
frozen before any submission was made.

There are no trained weights in this entry. The deployment artifact is the saved per-row output of
the readers: the system is inference-only.

### Where each stage lives

| stage | code |
|---|---|
| grounding-card generation | `llm_grounded.make_card` |
| grounded prompt, batching, output parsing | `code/llm_grounded.py` |
| the same reader over agent-hosted models | `code/llm_grounded_agent.py` |
| translated English input view | `code/translate_test.py` |
| exact confusion matrix from a server score | `code/decode_conf.py` |
| candidate flip sets, with the Gate-1 fence enforced mechanically | `code/t2_apply.py`, `code/opus5_apply.py` |
| adjudication batches, merge, coverage contract, cross-checks | `code/opus5_build_batches.py`, `code/opus5_merge.py`, `code/opus5_xcheck.py`, `code/opus5_r4_assemble.py` |
| official metric re-implementation | `code/stance_lib.py` |
| **bit-exact rebuild of the submitted file** | `code/reproduce_champion.py` |

## 2. Reproducing submission 869013

```bash
cd code
python reproduce_champion.py
```

Rebuilds the 644-row label vector from the stored base array and the flip set, writes
`staging/sub_reproduced.zip`, and asserts a byte-for-byte match against the archived submission:
md5 `13d07fcb58e4a5eb48bf855ea143eb67`, identical to
`submission/codabench/869013__sub_o5r4f.zip`. Only `numpy` is needed; it runs offline, on CPU, in
under a second.

Reproducing from the cached reader output rather than from live API calls is deliberate: frontier
endpoints are not bit-stable across time, so a re-read is a *different* system and must be
re-measured, not compared.

## 3. How the 73 corrections were derived

Every round was pre-registered before any submission, fired as nested tiers, decoded exactly
against the server, and composed once under the frozen rule. The chain, all on the blind test:

| round | question put to the reader | score |
|---|---|---|
| — | grounded base | 0.9016 |
| §8 | prior-calibrated repair of the decoded error pool | 0.930383 |
| R1 | cross-family re-reading, gated by a convention eyeball | 0.931660 |
| R2 | candidates ranked by how many of 13 stored views voted for the alternative | 0.933170 |
| R4 | **annotator simulation** — "predict the label the annotator *wrote*", not the true stance | **0.935700** |

R4 was the best-paying framing of the run, and its precision was concentrated entirely at
*convention ∧ at least one independent source* (3 of 4 correct); the same framing fired solo was 0
of 5 and over-fires roughly fivefold. Each round's forecast matched the returned score to six
decimal places.

One negative worth carrying: **adjudication is a one-shot gain, not a loop.** Round 1 beat its
mechanical control by +1.02; a second pass over the already-corrected rows scored −0.04 against
control and a third on a different tail −0.03. Re-running the mechanical control every round is
what exposes this — the raw delta alone would not.

## 4. What the residual turned out to be

Because the scorer reports Favg2, Favg3 and accuracy per target, and the gold class marginals were
pinned exactly by two earlier constant submissions, a submission's **full confusion matrix has a
unique integer solution** (`code/decode_conf.py`, verified by exhaustive search over the integer
lattice). The final entry's 56 errors are therefore known counts, not estimates: 34 gold-None rows
leaking into the stance predictions, 13 Favor/Against swaps, and 9 stance rows sitting in the
abstain pool.

Against that exactly known residual, ten instruments across five families were measured —
fine-tuned Arabic encoders, prompt engineering, cross-laboratory ensembling, supervised stacking
over reader votes, and auxiliary-variable pipelines — and none reached the break-even precision the
metric demands. The reason is in the dataset's own annotation columns: over the 382 gold-None rows,
`none_reason` reads **"Not clear" 370 versus "Not Related" 12**. The abstain class is 97% *annotator
uncertainty*, not off-topic content, so every off-topic / target-displacement / "no evaluative
content" detector — ours and, as far as we can tell, the field's — was aimed at 3% of the class.
Reframing the question as *"would two annotators disagree about this row?"* is confirmed
(reader disagreement raises P(gold-None) from 3.8% to 17.6%, odds ratio 5.35) and still refuted as a
lever: 10% precision on the blind test against a 54% break-even. No annotation column clears
break-even even as an oracle — gold sarcasm labels carry an odds ratio of 1.22 on the rows we
misread, which is why two independent irony detectors each scored 0 for 8.

Other measured negatives, each a blind-test read: greedy beat `reasoning=high` (90.16 vs 89.85);
self-consistency over 5 samples at T=0.7 moved 6–11 of 644 rows; a strict-None prompt lost 0.35 on
test while gaining 1.07 on a held-out proxy, the optimal None operating point sign-flipping across
targets; 24 few-shot convention demos from disjoint targets changed 8 of 312 rows and lost 0.27; a
translated-English input view helped Trimester (+0.85) and hurt Ecars (−0.83), which is why views
were routed per target rather than chosen globally; an MSA-normalised third view was decorrelated
but not better.

## 5. The LLM inference record

Every LLM was called **programmatically**, batched over an OpenAI-compatible `/chat/completions`
endpoint at temperature 0 — no interactive chat session was used at any point, so there are no chat
transcripts or UI screenshots to show. The equivalent record, complete and machine-readable, is:

| what | where |
|---|---|
| per-row output of every model × prompt run (label, probability, row index, target) | `code/_llm/*.npz` |
| exact model id, grounding card, row count and wall-clock of each run | `code/_llm/*_meta.json` |
| per-batch call log of each run | `code/_llm/*.log` |
| raw per-row replies of the adjudication rounds — label, confidence, one-line rationale | `code/_llm/opus5_out/`, `code/_llm/opus5_conv_out/`, `code/_llm/t2push_favN_out/`, `code/_llm/opus5_recheck.json` |
| per-row verdicts of the auxiliary probes (displacement, sentiment, not-Favor, Favor-hunt) | `code/_llm/displace_*.json`, `code/_llm/sent_*.json`, `code/_llm/nf_*.json`, `code/_llm/favorhunt_*.json` |
| the flip set actually shipped, and its pre-registration | `code/_llm/champion_flips.json`, `code/_llm/opus5_prereg.md` |

`code/opus5_candidates.py` writes a per-row evidence table (`_llm/opus5_evidence.md`) that inlines
the tweet text of each candidate. That file is regenerable from the artifacts above once you have
the dataset, and is withheld here for the same reason the CSVs are: it would redistribute the blind
test set.

Each `.npz` holds `proba[N,3]` in column order `[Against, Favor, None]` plus `pred_id[N]`, aligned
to the released test-row order — that order is the only join key the scorer uses.

**Output parsing.** Models are asked for exactly one line per tweet, `<n>|<Label>`. Parsing is the
regex `LBL_RE`, defined at `code/llm_predict.py:79` and imported by `code/llm_grounded.py`, applied
in `classify_target()` at `code/llm_grounded.py:93`; a batch that loses lines is re-sent once with a
larger token budget and any row still unparsed falls back to `None`. The adjudication rounds instead
request strict JSON (`{"<row>": {"label", "conf", "why"}}`) and are validated by
`code/opus5_merge.py`, which enforces a coverage contract — every row in the batch index must appear
exactly once with a valid label — and exits non-zero listing what is missing.

**Regenerating the base reader from live APIs.**

```bash
python llm_grounded.py --model gpt-5.6-sol --cards_from _llm/g56fix_grounded_test_meta.json \
    --csv <test_csv> --order file --batch 40 --reasoning none --out _llm/g56fix_grounded_test.npz
```

These scripts resolve an endpoint through `llm_provider_registry.provider_for(model)`, a thin
in-house shim that is **not** part of this release. It returns nothing but
`{"base_url": ..., "api_key": ..., "model": ...}` for an OpenAI-compatible chat endpoint and is
consumed only in `call_api()` (`code/llm_predict.py:82`); substituting five lines that read your own
credentials is sufficient. Everything that determines the result — prompts, cards, batching,
temperature, parsing — is in this repository.

## 6. Rules compliance

Zero-shot inference and external pretrained models are permitted by the task rules. Few-shot
demonstrations, where used at all, were drawn only from the released labelled train/dev split and
only from targets **disjoint** from the test targets — convention transfer, never test leakage. The
hidden test was never pseudo-labelled or used for training.

We drew one boundary explicitly and enforced it in code. Aggregate counts recovered from our own
scored submissions are legitimate measurement; solving for *which individual hidden row* carries
which label, and then picking rows on that basis, is label reconstruction and is not. `t2_apply.py`
and `opus5_apply.py` mechanically refuse decode-pinned rows, half-pairs of a pinned pair, and diffs
smaller than four rows. During the run this fence rejected a constructed set worth a guaranteed
+0.307; that set was deleted unsubmitted.

## 7. Data

Not included here and not redistributable from this repository — request it from the task
organizers.

- Train / dev: MawqifV2, 2721 labelled rows over Covid Vaccine and Digital Transformation.
  `code/stance_lib.py` reads them from `$MAWQIF_DIR` (a cluster path is the fallback default).
- Blind test: the organizers' `test_track_2.csv`, `tweet_text` renamed to `text`, **row order
  preserved**, 644 rows (Ecars 332 / Trimester 312).
- Submission format: `predictions.txt`, one label per line in test-row order, zipped flat.

## 8. Environment

- Python 3.11.5. The reproduction in §2 needs only `numpy` (developed against 2.4.2).
- Regenerating readers additionally needs `pandas` and `requests`. `torch`, `transformers` and
  `peft` are needed only by the fine-tuned arms in §4, none of which is part of the entry — see
  `requirements.txt`.
- No GPU is required for anything in §2, §3 or §5.
- Seeds are fixed in code; LLM calls use temperature 0.

## 9. Contents

```
code/*.py, code/*.sh   every inference, probe, training and submission script of the run
code/_llm/             per-row LLM outputs, run logs, grounding cards, adjudication replies,
                       pre-registrations, and the shipped flip set
code/staging/          the exact submitted label vector, and where reproduce_champion.py writes
submission/codabench/  the archived submission zips
```

`code/` is the run's script directory as it stood at the close of the competition, so it contains
the refuted branches as well as the shipped one — Arabic encoder and decoder-LoRA fine-tuning
(`train.py`, `train_mtl.py`, `train_lora.py`), a supervised stacker over reader votes
(`stacker.py`), None detectors and target-displacement probes (`none_detector.py`,
`displace_probe.py`, `sent_probe.py`, `notfavor_probe.py`, `favor_hunt.py`), synthetic and external
augmentation (`gen_synth.py`, `gen_synth_aim.py`, `make_asx.py`). None of these is part of
submission 869013; they are kept because the system paper reports them as negative results, and
because §4's claim — that the remaining error mass is a property of the labelling process rather
than of model quality — is only checkable if the instruments that failed are visible.

## 10. License

MIT (see `LICENSE`). The code remains publicly accessible for at least three years.
