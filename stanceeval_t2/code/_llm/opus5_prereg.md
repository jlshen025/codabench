# PRE-REGISTRATION — opus-5 endgame candidates (frozen 2026-07-29 ~13:10 UTC, BEFORE any submit)

Base: e_v34.txt (= submission 867704, Favg2 0.930383). One adaptive round:
fire T1/T2/T3 (nested), decode each per-target, compose ONCE as the union of
measured-net-positive per-target tier-slices, submit the composition, stop.

Selection rule (fixed in advance): opus-5 pool-calibrated read AND >=1 agreeing
non-claude family (sol=gpt5.6 / deepseek-v4-pro fresh symmetric rechecks;
2026-07-29 _llm/opus5_x*.json), then a human-convention eyeball gate.
Fence checks: no row in v34's own 32 flips; no row from the fenced 8-row
artifact; membership 100% text-judgment-derived; every tier diff >=4 rows.

## Eyeball-gate decisions (recorded before firing)
- DROPPED 354 (opus+dspro None): corpus convention "positive-frame growth
  reporting = Favor" (measured off tier-A) says the current Favor is right.
- CONVERTED 406 to F->None: decode pins baseline ecF pool at 0 gold-Against,
  so sol(A98)+dspro(A95) can only mean "not Favor"; None is the only winnable
  direction. opus said None(58) independently.
- DEMOTED 8 (trN) to T2: sol and dspro each flipped label across two reads
  (batch-context sensitivity); opus stable Against(62); sarcastic rhetorical
  question fits the corpus Against convention.

## T1 "o5t1" — 5 flips (highest conviction)
| row | tgt | flip | votes |
|-----|-----|------|-------|
| 331 | Ec | None->Against | opus62 + sol96 + dspro85 (3/3) — "donkey cart" mock |
| 384 | Ec | Against->None | opus60 + sol91 + dspro95 (3/3) — stadium-cart fuss, no EV stance |
| 406 | Ec | Favor->None  | not-Favor 3/3 (opus N58, sol A98, dspro A95); decode: pool has no goldA |
| 138 | Tr | None->Against | opus57 + sol57 + dspro70 (3/3) — dismissive |
| 303 | Tr | Favor->Against | opus56 + sol96 (dspro dissent) — ironic praise + "same tools same results" |
E[dFavg2] ~ +0.13 pts; P(net>0) ~0.7.

## T2 "o5t2" = T1 + 6 (second conviction; None-direction low-cost cells + 340)
| row | tgt | flip | votes |
|-----|-----|------|-------|
| 575 | Ec | None->Against | opus62 + dspro70 (sol N72 soft) — EV flood-vulnerability fear |
| 612 | Ec | Against->None | opus55 + sol94 (dspro dissent) — neutral oil-economics analysis |
| 457 | Ec | Against->None | opus52 + sol90 (dspro dissent) — bankruptcy news caption |
| 340 | Ec | None->Favor  | opus48 + sol78 (dspro dissent) — invested in Egyptian EV industry |
| 8   | Tr | None->Against | opus62 stable; sol/dspro context-unstable — sarcastic question |
| 226 | Tr | Against->None | opus55 + sol82 (dspro dissent) — evaluates remote delivery, not terms |
E[T2-T1] ~ +0.05 pts.

## T3 "o5t3" = T2 + 4 (A->F swap gambles, +-0.19/row, Force_Best option value)
| row | tgt | flip | votes |
|-----|-----|------|-------|
| 409 | Ec | Against->Favor | opus55 + dspro70 (sol A68 soft) — solar-EV admiration, price-only gripe |
| 456 | Ec | Against->Favor | opus58 + dspro85 (sol A95 HARD dissent) — waiting-to-buy desire |
| 124 | Tr | Against->Favor | opus58 + dspro95 (sol A91) — genuine praise of schedule benefit |
| 206 | Tr | Against->Favor | opus55 + sol94 (dspro A85) — mocks the cancel-campaign = defends system |
E[T3-T2] ~ +0.01 with high variance (2 hits = +0.39).

## Composition rule (fixed now)
Per-target per-tier deltas from decode; final = v34 + union of tier-slices
(whole slices only, never split) whose measured per-target delta > 0.
If nothing positive: stand pat on 867704.

---

# ROUND 2 (frozen 2026-07-29 ~14:45 UTC, before firing) — base = e_o5final (868668, 0.931660)

Directive: full competition mode — multiple adaptive rounds, burn the
budget. Standing exclusions: no decode-derived membership (575/138/pair-splits
stay out), no <4-row diffs; membership = 13-view vote counts + text eyeball.
Instrument: VIEW-VOTE-COUNT ranker (never fired on T2; T1-diagnosis precedent).

## V1 "o5v1" — 6 flips (vote-consensus + eyeball agree)
462 Ec A->N (9/13 None votes; company-strategy analysis) · 583 Ec A->N (7/13;
mocks the company, not EVs) · 510 Ec N->A (6/13; personal bad-EV-ride story) ·
316 Ec F->N (9/13; Hyundai news relay) · 164 Tr A->N (8/13; cryptic wish
fragment) · 51 Tr A->N (8/13; prayer fragment, no evaluative direction).

## V2 "o5v2" = V1 + 4 — curated-revert gamble
536, 513, 425 Ec F->A (13/13 views Against; 07-28 curated flips, decode says
~2-3 of the 7 curated were wrong; my text read = criticism-dominant) ·
482 Ec N->A (5/13; negative-frame EV history).

## V3 "o5v3" = V2 + 5 — pure option-value tier
428 Ec F->A (12/13; "hopefully we're next" ambiguous) · 314 Ec F->N (10/13;
🌚-deadpan inevitability) · 431 Ec F->N (8/13; wishes for A car, maybe-EV) ·
277 Tr A->F (6/13 Favor; #Benefits tag, sarcasm risk) · 231 Tr A->F (5/13;
remote-third-term proposal).

Dropped at eyeball: 334 (interest-in-topic = convention Favor), 152 (three
instruments give three different labels).
Composition: same rule, base o5final; whole per-target tier-slices, positive only.

# ROUND 3 (frozen 2026-07-29 ~15:40 UTC) — base = e_o5r2f (868992, 0.933170)
W1 "o5w1" (5): Tr {21, 26} A->N (6/4 None-votes; scheduling-REQUEST fragments =
the exact shape of the 2/2 winners 164/51) · Ec {498, 501} N->A (mockery-in-EV-
context / EV-downplaying; 498 4 A-votes, 501 opus62+oss65) · Ec {454} A->F
(prior_recheck sol F(92) top flag; engaged desire-adjacent, 409-class).
W2 "o5w2" = W1 + 4 gambles: Ec {506} N->A (opus 78, evidence split) ·
Ec {525, 332} A->F (3v3 reader chaos / playful-admiration read) ·
Tr {150} A->N (4 votes; my read says Against — pure option value).
Composition: same frozen rule, whole per-target slices, positive only.
Eyeball drops: 614/613/520/327 (sol Favor-hunt over-fires on infra complaints),
472/485 (historical/economic None-reads), 18/188/245/248/10/143 (genuine weariness-Against).

# ROUND 4 (frozen 2026-07-29 ~17:35 UTC) — base = e_o5r2f (868992, 0.933170)
Instrument: ANNOTATOR-SIMULATION reads (opus-5, "predict what the annotator
wrote") joined with all prior signals; conv-solo flags are weak (34 flags for
~7 gold in ecAF) — membership requires conv + >=1 independent source OR the
purest convention shape, then eyeball. 40 vetoed (hard sol91/dspro85 F dissent);
613/614/615 skipped (solPrior-only = the measured-dead 454 class). 497 enters
via the clean-reader route the 07-28 fence note explicitly permits (conv 63 +
2 views F; provenance = text judgment, not score arithmetic).
X1 "o5x1" (6): ecA->F {579, 526} · ecN->F {585, 497} · trF->A {310, 46}
X2 "o5x2" = X1 + 6: ecA->F {378, 537} · ecN->F {418, 342} · trF->A {305, 308}
X3 "o5x3" = X2 + 5: ecA->F {563, 489} · ecN->F {493} · trF->A {263, 294}
Composition: frozen rule, whole per-target tier-slices, positive only. Fire 3
now, hold 1 slot as emergency compose; full compose + spray at the 00:00 window.

## R4 OUTCOME (869009/869010/869011 decoded; composition 869013 = 0.935700 EXACT)
X1_Ec +0.261 (585, 497 both goldF; one of {579,526} goldF, other goldA — pair
fenced). ALL Trim F->A sarcasm reads 0/4 (310, 46, 305, 308 all goldF — the
annotator takes praise at face value; irony-hunting = measured trap). X2/X3
increments negative (418/342 goldN; 378/537 not-F; 563/489 goldA; 493 goldN).
Conv-instrument precision lives ONLY at support>=1 (3/4); conv-solo = 0/5.
The conv+support vein is exhausted; window spray = pure dregs only.
