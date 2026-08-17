"""grounded_prompts.py — the two target-grounded prompt builders from stanceeval_t2's
llm_grounded.py, extracted verbatim so llm_grounded_agent.py has ZERO dependency on
t2's llm_predict._provider_for (which this project's llm_predict.py does not define).

Both are fully target-parameterised: sys_prompt(target, card, strict_none) takes the
target as an argument, so nothing here is Track-2 specific.
Provenance: stanceeval_t2/scripts/llm_grounded.py, copied 2026-07-29.
"""

CARD_SYS = ("You are an expert in Arabic public discourse. Given a stance-detection TARGET, "
            "output ONLY a compact JSON object with keys 'definition' (one concise Arabic sentence "
            "defining the target) and 'aspects' (6-8 comma-separated Arabic sub-aspects / paraphrases / "
            "consequences that people express support or opposition about). No other text.")


def sys_prompt(target, card, strict_none=False):
    lines = ["You are an expert annotator for Arabic stance detection. Classify the AUTHOR's stance "
             "TOWARD THE TARGET as exactly one of: Favor, Against, None.",
             f"TARGET: {target}"]
    if card.get("definition"):
        lines.append(f"MEANING: {card['definition']}")
    if card.get("aspects"):
        lines.append(f"RELATED ASPECTS: {card['aspects']}")
    none_line = ("- None: use ONLY if the tweet is unrelated to the target, or purely factual/descriptive "
                 "with no evaluative stance. Do NOT default to None merely because the target is not named literally.")
    if strict_none:
        none_line = ("- None: assign ONLY when the tweet is genuinely off-topic OR purely factual/descriptive with "
                     "ZERO evaluative content. If the author reveals ANY leaning — even faint, indirect, via a related "
                     "aspect/consequence, sarcasm, rhetorical question, or emotional tone — you MUST choose Favor or "
                     "Against, NEVER None. 'None' means truly no stance, NOT 'uncertain'. When genuinely torn between "
                     "None and a stance, choose the stance.")
    lines += [
        "Guidelines:",
        "- Stance may be expressed INDIRECTLY via a related aspect or consequence — that still counts as stance toward the target.",
        "- Favor: the author supports, praises, endorses, defends, or is glad about the target or its aspects.",
        "- Against: the author opposes, criticizes, distrusts, mocks, fears, or rejects the target or its aspects. Sarcasm and rhetorical questions usually signal Against.",
        none_line,
        "Output EXACTLY one line per tweet formatted '<number>|<Label>' (Label = Favor, Against, or None). No other text.",
    ]
    return "\n".join(lines)
