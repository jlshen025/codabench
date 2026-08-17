#!/usr/bin/env python
"""Build pool-targeted bilingual batch files for the opus-5 recheck pass.

Pools are defined by the champion vector e_v34.txt; priors are the EXACT
decoded confusion at 867704 (aggregate counts only — Gate-1 clean).
v34's own flip rows (21 Ecars None->F, 11 Trim None->A) are EXCLUDED
(re-adjudicating our own adjudicated output measured regressive).
Output: _llm/opus5_batches/<pool>_<n>.txt + _llm/opus5_batches/index.json
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AR = pd.read_csv(HERE / "test_norm.csv", keep_default_na=False, encoding="utf-8-sig")
EN = pd.read_csv(HERE / "_llm" / "test_en.csv", keep_default_na=False, encoding="utf-8-sig")
PRED = [l.strip() for l in (HERE / "staging" / "e_v34.txt").read_text().splitlines() if l.strip()]
CARDS = json.load(open(HERE / "_llm" / "fixed_cards.json"))["cards"]
assert len(AR) == len(EN) == len(PRED) == 644

FLIPPED_EC = {320, 345, 360, 363, 391, 394, 398, 424, 437, 441, 464, 480, 492,
              495, 505, 534, 542, 551, 570, 574, 581}          # v34 None->Favor
FLIPPED_TR = {56, 66, 92, 141, 144, 158, 173, 211, 219, 238, 242}  # v34 None->Against

# pool -> (target, v34 label, prior paragraph)
POOLS = {
    "ecN": ("Ecars", "None",
            "All rows below are currently labeled None by a strong, heavily-validated ensemble. "
            "Measured against the hidden gold (aggregate counts only): about 4 of these 52 rows are actually Favor "
            "and about 4 are actually Against; the other ~44 are correctly None. Find the ~8 mislabeled rows; "
            "agree with None everywhere else. Precision matters more than recall — only call a stance when the text supports it."),
    "trN": ("Trimester", "None",
            "All rows below are currently labeled None by a strong ensemble. Measured against the hidden gold: "
            "about 2 of these 14 rows are actually Against and 0 are Favor; the other ~12 are correctly None. "
            "Find the ~2 hidden Against rows; agree with None everywhere else."),
    "ecA": ("Ecars", "Against",
            "All rows below are currently labeled Against by a strong ensemble. Measured against the hidden gold: "
            "about 8 of these 125 rows are actually Favor and about 5 are actually None; the other ~112 are correctly Against. "
            "The hidden Favor rows are the big prize. Watch for: sarcasm mis-read as literal, complaints about PRICE or "
            "infrastructure from someone who clearly WANTS the car (desire = Favor), quoting critics to rebut them, "
            "and conditional praise. Only flip when the text genuinely supports it."),
    "trA": ("Trimester", "Against",
            "All rows below are currently labeled Against by a strong ensemble. Measured against the hidden gold: "
            "about 3 of these 217 rows are actually Favor and about 6 are actually None; the other ~208 are correctly Against. "
            "Most complaints about exhaustion/pressure under the three-term system are genuine Against — do not flip those. "
            "Look for: rows defending the system or attacking its critics (Favor), and pure news/announcements or "
            "off-topic rows with no evaluative stance (None)."),
    "ecF": ("Ecars", "Favor",
            "All rows below are currently labeled Favor by a strong ensemble. Measured against the hidden gold: "
            "about 14 of these 134 rows are actually None and about 0 are Against; the other ~120 are correctly Favor. "
            "The annotators were LIBERAL: purchase intent, admiration, wishing for one, recommending — all genuine Favor, keep them. "
            "The hidden None rows are pure news/ads/factual reports with NO evaluative lean by the AUTHOR "
            "(e.g. retweeted headlines, price listings, neutral questions). Only call None when there is truly no stance."),
    "trF": ("Trimester", "Favor",
            "All rows below are currently labeled Favor by a strong ensemble. Measured against the hidden gold: "
            "about 2 of these 70 rows are actually Against and 0 are None; the other ~68 are correctly Favor. "
            "Look for sarcasm or ironic 'praise' that actually mocks the three-term system."),
}

BATCH = 22
OUTDIR = HERE / "_llm" / "opus5_batches"
OUTDIR.mkdir(exist_ok=True)


def header(pool_key: str) -> str:
    target, _, prior = POOLS[pool_key]
    card = CARDS[target]
    return "\n".join([
        "You are an expert annotator for Arabic stance detection (Saudi-dialect tweets). "
        "Classify the AUTHOR's stance TOWARD THE TARGET as exactly one of: Favor, Against, None.",
        f"TARGET: {target}",
        f"MEANING: {card['definition']}",
        f"RELATED ASPECTS: {card['aspects']}",
        "Guidelines:",
        "- Stance may be expressed INDIRECTLY via a related aspect or consequence — that still counts as stance toward the target.",
        "- Favor: the author supports, praises, endorses, defends, or is glad about the target or its aspects.",
        "- Against: the author opposes, criticizes, distrusts, mocks, fears, or rejects the target or its aspects. "
        "Sarcasm and rhetorical questions usually signal Against.",
        "- None: use ONLY if the tweet is unrelated to the target, or purely factual/descriptive with no evaluative stance. "
        "Do NOT default to None merely because the target is not named literally.",
        "- The EN line is a machine translation for convenience; the AR line is authoritative when they disagree.",
        "",
        "CONTEXT FOR THIS PASS: " + prior,
        "",
        'Output: ONLY a JSON object mapping each row index (as a string) to '
        '{"label": "Favor|Against|None", "conf": <0-100>, "why": "<one short clause>"}. '
        "Every row below must appear exactly once. No other text.",
        "",
    ])


def row_block(i: int) -> str:
    return f"{i}| AR: {AR.text[i]}\n   EN: {EN.text[i]}"


index = {}
n_files = 0
for key, (target, lab, _) in POOLS.items():
    rows = [i for i in range(644)
            if AR.target[i] == target and PRED[i] == lab
            and i not in (FLIPPED_EC | FLIPPED_TR)]
    index[key] = rows
    for b in range(0, len(rows), BATCH):
        chunk = rows[b:b + BATCH]
        p = OUTDIR / f"{key}_{b // BATCH:02d}.txt"
        p.write_text(header(key) + "\n\n".join(row_block(i) for i in chunk) + "\n")
        n_files += 1

json.dump(index, open(OUTDIR / "index.json", "w"), indent=1)
sizes = {k: len(v) for k, v in index.items()}
print(f"pools: {sizes}  total rows {sum(sizes.values())}  batch files {n_files}")
