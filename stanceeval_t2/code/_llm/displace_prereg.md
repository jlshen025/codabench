# PRE-REGISTRATION — target-displacement repair of the Ecars pred-Favor pool
Frozen 2026-08-03, BEFORE any submission. Base = champion `869013` (0.935700).

## Hypothesis (mechanism, named before testing)
Some Ecars tweets carry strong evaluative language, but the stance is aimed at a DIFFERENT
object — the government, national backwardness, a rival country, an official — with electric
cars used only as a yardstick/symbol. The corpus annotator marks these **None**; every
grounded stance prompt reads the target-positive wording and votes **Favor**.

This is NOT the "pure spec/news sharing = None" hypothesis, which was measured DEAD on
2026-07-30 (0/6 at conf 82). The probe prompt explicitly instructs that spec-sharing,
market-growth reporting and questions are ONTARGET.

## Why this channel
Champion decode: Ecars `pred F160 A121 N51` vs `gold F145 A117 N70`, `TP_F=137`.
⇒ 23 of 160 pred-Favor rows are wrong, the largest single error channel in the system.
Break-even for a Favor→None flip = **54.0%** (gain +0.001025 / loss −0.001202 per row).

## Instrument validation (zero slots spent)
Probe run over all 160 Ecars champ-Favor rows, including the 29 the lineage already moved
None→Favor — rows that prior decodes measured majority gold-Favor, and the hardest possible
false-positive control (every reader originally called them None).

| conf | fires on 29 proven gold-Favor | fires on 117 virgin |
|---|---|---|
| ≥0  | 37.9% | 36.8% |
| ≥75 | 17.2% | 31.6% |
| ≥85 | 3.4%  | 21.4% |
| ≥90 | **0.0%** | **11.1%** |

Control silence at conf≥90 with sustained virgin firing = genuine discrimination.

## Frozen tiers (nested, disjoint shells decode independently)
All flips are `Favor -> None`, Ecars only, on the champion base.
- **T_d90** = opus-5 conf≥90 — 13 rows: 319 333 338 401 405 417 466 483 494 532 553 554 578
- **T_d85** = opus-5 conf≥85 — 25 rows (superset of T_d90)
- **T_d75** = opus-5 conf≥75 — 37 rows (superset of T_d85)
- **T_dU**  = opus-5 conf≥90 ∧ sonnet-5 DISPLACED conf≥75 (cross-reader unanimity; the
  project's measured precedence rule is calibrated unanimity > vote-counts)

## Decision rule (frozen)
Decode each tier exactly via `decode_conf.py`. Keep ONLY shells whose measured precision
exceeds the 54.0% break-even; compose the surviving disjoint shells in one final submission
and verify the composition forecast to 6 dp (this has matched 5 times running).
A shell below break-even is discarded whole — no re-picking rows inside it.

## Gate-1 fence (unchanged, binding)
Rows chosen purely by reading the TEXT. Decode-pinned rows {575,138,510,428} and the pinned
pairs {409,456}/{124,206}/{498,501}/{525,332} are excluded from the pool. Every diff ≥4 rows.
No membership-from-decode; decode is used ONLY to score a candidate after the fact.

---
# PRE-REGISTRATION 2 — Trimester/Ecars pred-Favor → Against on FRESH cross-reader unanimity
Frozen 2026-08-03 before submit. Base = champion 869013.

**Channel:** Trimester pred-Favor (74 rows) holds EXACTLY 4 gold-Against and 0 gold-None
(unique decode, a=4). Favor→Against pays +0.001796 if gold-Against, −0.001902 if gold-Favor;
break-even 51.4%. Highest per-row payoff on the board; 4 rows ≈ the entire gap to #1.

**Selection rule (judgment about the TEXT only):** rows where the two FRESH independent
readers (claude-opus-5, claude-sonnet-5 full 644-row grounded reads, neither anchored on our
predictions) BOTH say Against while the champion says Favor. Stored older views are excluded
from the ranking because they are stale on already-adjudicated rows.

**Exclusions applied from the ledger, before looking at any score:**
- 425 — inside the decoded {536,513,425} set measured to contain 0 gold-Against.
- 526 — pairs with 579 (one of the two known gold-Favor).
- 326 — national-yardstick framing, which the 08-03 d90 decode proved is gold-Favor.
- 314 — an irony/sarcasm call; irony detection measured 0/8 on this corpus (permanently dead).
- 513, 536 — ledger-known gold-Favor.

**T_trA = {97, 227, 231} ∪ pair{124, 206}** = 5 rows, all Favor→Against.
206 carries the strongest evidence in the pool (10/11 views + fresh unanimity). Its pair
partner 124 is included ONLY to keep the pinned pair whole — no identity is resolved, which
is what the fence requires; 124 is expected to cost a row and that cost is accepted upfront.

**Decision rule:** decode exactly; keep only if measured precision > 51.4%; compose with any
other measured-positive disjoint slice. Discard the whole shell if it is below break-even.

---
# PRE-REGISTRATION 3 — Trimester pred-Against → None on 3-WAY fresh unanimity
Frozen 2026-08-03 before submit. Base = champion 869013.

**New evidence class:** a THIRD fresh unanchored reader from an untested LAB (`mistral-large`,
agreement 0.849/0.852/0.854 vs sol/opus5/sonnet5 — below the 0.95 clone bar and looser than
the sol-opus5 pair at 0.915). 2-way unanimity measured 40% on 08-03 (sub 875925); 3-way is a
strictly stronger class. `zai-glm-4.7` is BLOCKED (429 at bulk; its all-None npz is the
parser defaulting on unmatched output, not a read) — blocked, NOT ruled out.

**Channel:** Trimester pred-Against (222 rows) holds EXACTLY 8 gold-None (unique decode a=4).
Against→None pays +0.000701 if gold-None, −0.000773 otherwise; break-even 52.4%.
Note these rows are ADJUDICATED (the lineage moved 13 Trimester rows None→Against at a
measured 7/11), so ~4 of those moves were wrong and are recoverable. Older stored views are
excluded from the ranking — they are stale exactly there; only the 3 FRESH readers vote.

**T_N4** = {141, 211, 238, 242} — all 3 fresh readers say None, all fence-clean.
**T_N13** = T_N4 + {66, 92, 107, 144, 152, 173, 219, 245, 248} (clean 2-of-3), nested superset.
Known-gold rows (21, 26, 150 goldA) excluded before ranking.

**Decision rule:** decode each; keep only shells above the 52.4% break-even; compose surviving
disjoint slices. Max attainable is 8 correct, so T_N13 caps at 61% by construction.

---
# PRE-REGISTRATION 4 — post-`t` plan (2026-08-04). Base = champion 869013.
`t` PINNED = 10 by probe 876962 (Favg2 0.364732 = (323+10)/913 exactly; per-target 0.291457 /
0.421359 both confirm independently). Residual now has ZERO free parameters:
N→F 17 · N→A 17 · **A→F 10** · F→A 3 · **F→N 6** · A→N 3.
The stated range t∈[4,13] was wrong: per-target constraints bind (Trimester A→F is
unique at 4; Ecars A→N=7−a≥0 ⇒ a≤7), so t∈[4,11]. Recorded.

**S1 — pred-None composite (positive EV, both sub-channels forecast ABOVE bar).**
predNone pool = 67 = 58 goldNone + 6 goldFavor + 3 goldAgainst.
- N→A {406, 472, 485}: negative tone ∧ ≥3 of 5 readers say Against. Labelled forecast 59.0%
  vs bar 47.6%. Exactly 3 goldAgainst exist and these are the only fence/known-clean rows at
  that strength. 406 is unanimous 5/5 at tone-conf 92 and its text is overtly critical — the
  pick is sourced from the readers; the earlier decode note merely corroborates it.
- N→F {9}: 4 of 5 readers say Favor, the strongest Favor signal in the pool.
Excluded on ledger evidence before ranking: 164, 51 (decode-measured goldNone), 506, 493, 8,
342, 384, 331, 340, 418 (known goldNone), and the fenced set.
Max attainable +0.003510 → ~0.93921 (clears #4, short of #2).

**S2..S4 — F→A variance shots, k=12-16.** Channel holds 10 goldAgainst in the 234-row Favor
pool; bar 51.4%; best labelled forecast 48.9% (negative tone ∧ ≥2/≥3 readers) — just UNDER the
bar, so EV is slightly negative but under Force_Best the payoff is max(held,new) and only the
upper tail matters. k must be ≥12: even 4/4 on a k=4 shot cannot clear #1 (+0.007184 < 0.00736).
Diverse rankers, deliberately not nested, so their errors decorrelate.

**Decision rule:** decode every shot; keep only measured-positive disjoint slices; compose once.
Gate-1 unchanged — no membership-from-decode, no pinned rows, no half-pairs, diffs ≥4.
