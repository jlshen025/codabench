"""
llm_grounded.py — TARGET-GROUNDED Arabic stance voter (claude), for UNSEEN-target transfer.

Highest-EV lever per two independent model consults: instead of classifying against the
raw target string, give the LLM a per-target "card" (auto-generated 1-line Arabic definition
+ related aspects/paraphrases) so it can map ABSTRACT targets (e.g. Digital Transformation)
to their semantic neighborhood, and count INDIRECT stance via an aspect. The card is
AUTO-GENERATED per target → the METHOD is target-agnostic → generalizes to unknown test
targets (no per-target hand-tuning = Gate-2 safe). Also biases away from None (excluded
from Favg2). Reuses llm_predict's provider + API + label regex + metric.

Runs on LOGIN (internet). Output npz matches llm_predict (proba/pred_id/target/idx/y_true)
for drop-in ensembling. Cards cached to <out_dir>/cards.json.
"""
import os, sys, json, argparse, time, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from llm_predict import _provider_for, call_api, LBL_RE

CARD_SYS = ("You are an expert in Arabic public discourse. Given a stance-detection TARGET, "
            "output ONLY a compact JSON object with keys 'definition' (one concise Arabic sentence "
            "defining the target) and 'aspects' (6-8 comma-separated Arabic sub-aspects / paraphrases / "
            "consequences that people express support or opposition about). No other text.")


def make_card(cfg, model, target):
    msgs = [{"role": "system", "content": CARD_SYS},
            {"role": "user", "content": f"TARGET: {target}"}]
    try:
        txt = call_api(cfg, model, msgs, "none", 2000, 0.0)  # generous: reasoning models eat tokens before JSON
        m = re.search(r"\{.*\}", txt, re.S)
        d = json.loads(m.group(0)) if m else {}
        defn = str(d.get("definition", "")).strip()
        asp = d.get("aspects", "")
        asp = ", ".join(asp) if isinstance(asp, list) else str(asp).strip()
        if defn:
            return {"definition": defn, "aspects": asp}
    except Exception as e:
        print(f"[card] FAILED for {target!r}: {e}", flush=True)
    return {"definition": "", "aspects": ""}


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


def build_demos(k_per_class, seed=0):
    """Few-shot demos teaching the ANNOTATION CONVENTION (not the target): sampled from the
    LABELED pool (train Covid/Digital + dev WomenEmp) — all targets DISJOINT from the test
    targets, so this is convention transfer, never target leakage. Deterministic by seed."""
    import stance_lib as S
    pool = []
    for csv in (S.TRAIN_CSV, S.DEV_CSV):
        d = pd.read_csv(csv, keep_default_na=False, dtype=str)
        for _, r in d.iterrows():
            t, tg, st = str(r["text"]).strip(), str(r["target"]).strip(), str(r["stance"]).strip()
            if st in S.LABEL2ID and 40 < len(t) < 240:
                pool.append((tg, t, st))
    rng = np.random.default_rng(seed)
    out = []
    for lab in ("Favor", "Against", "None"):
        cand = [p for p in pool if p[2] == lab]
        for j in rng.choice(len(cand), min(k_per_class, len(cand)), replace=False):
            out.append(cand[int(j)])
    rng.shuffle(out)
    return "\n".join(f"[TARGET: {tg}] {t}\n-> {st}" for tg, t, st in out)


def classify_target(rows_idx, texts, cfg, model, sysp, batch, reasoning, temperature=0.0, throttle=0.0):
    """Return proba [n,3] (one-hot) for the tweets of ONE target."""
    proba = np.zeros((len(texts), 3), dtype=float)
    for b0 in range(0, len(texts), batch):
        if throttle and b0:
            time.sleep(throttle)  # stay under rate-limited relays (cerebras/groq)
        chunk = texts[b0:b0 + batch]
        user = "Classify each tweet's stance toward the TARGET:\n\n" + "\n".join(
            f"{i+1}| {t}" for i, t in enumerate(chunk))
        content = call_api(cfg, model, [{"role": "system", "content": sysp},
                                        {"role": "user", "content": user}], reasoning, 100 * len(chunk) + 1000, temperature)
        got = {}
        for line in content.splitlines():
            mm = LBL_RE.match(line)
            if mm:
                got[int(mm.group(1))] = mm.group(2).capitalize()
        miss = [k for k in range(1, len(chunk) + 1) if k not in got]
        if len(miss) > max(2, len(chunk) // 5):
            content = call_api(cfg, model, [{"role": "system", "content": sysp},
                                            {"role": "user", "content": user}], reasoning, 40 * len(chunk) + 300, temperature)
            for line in content.splitlines():
                mm = LBL_RE.match(line)
                if mm:
                    got.setdefault(int(mm.group(1)), mm.group(2).capitalize())
        for j in range(len(chunk)):
            proba[b0 + j, S.LABEL2ID.get(got.get(j + 1, "None"), 2)] += 1.0
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4.5")
    ap.add_argument("--csv", default=S.DEV_CSV)
    ap.add_argument("--order", default="file", choices=["loto", "file"])
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cards_from", default="", help="load fixed cards (meta JSON with 'cards') instead of generating")
    ap.add_argument("--provider", default="", help="force a specific relay leg (e.g. sub2apigpt for luna/terra)")
    ap.add_argument("--temperature", type=float, default=0.0, help=">0 enables stochastic sampling (self-consistency across runs)")
    ap.add_argument("--strict_none", action="store_true", help="aggressive anti-None guideline (metric-aware: Favg2 excludes None)")
    ap.add_argument("--throttle", type=float, default=0.0, help="sleep seconds between batches (rate-limited relays)")
    ap.add_argument("--demos", type=int, default=0, help="RAG few-shot: N labeled demos PER CLASS (convention transfer; targets disjoint from test)")
    args = ap.parse_args()
    pname, cfg = _provider_for(args.model, args.provider); wire = cfg["wire_model"]
    print(f"[provider] {pname} wire={wire}", flush=True)

    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    labeled = "stance" in df.columns and (df["stance"].str.strip() != "").all()
    if args.order == "loto":  # group by target (matches encoder LOTO OOF + llm_predict loto order)
        order_idx = []
        for tg in dict.fromkeys(df["target"].tolist()):
            order_idx.extend(df.index[df["target"] == tg].tolist())
    else:
        order_idx = df.index.tolist()
    targets = list(dict.fromkeys(df.loc[order_idx, "target"].tolist()))

    # one card per unique target: load fixed cards if given, else generate
    cards = {}
    t0 = time.time()
    _fixed = json.load(open(args.cards_from)).get("cards", {}) if args.cards_from else {}
    for tg in targets:
        cards[tg] = _fixed.get(tg) or make_card(cfg, wire, tg)
        print(f"[card] {tg!r}: def={cards[tg]['definition'][:60]!r} asp={cards[tg]['aspects'][:60]!r}", flush=True)

    global DEMOS
    DEMOS = build_demos(args.demos) if args.demos else ""
    if DEMOS: print(f"[demos] {DEMOS.count('->')} examples", flush=True)
    proba = np.zeros((len(order_idx), 3), dtype=float)
    pos_of = {gi: k for k, gi in enumerate(order_idx)}
    for tg in targets:
        idxs = [i for i in order_idx if df.at[i, "target"] == tg]
        texts = [df.at[i, "text"] for i in idxs]
        _sp = sys_prompt(tg, cards[tg], args.strict_none)
        if args.demos:
            _sp += ("\nAnnotated EXAMPLES from other targets — copy their labelling convention "
                    "(how much implicitness still counts as a stance):\n" + DEMOS)
        pr = classify_target(idxs, texts, cfg, wire, _sp, args.batch, args.reasoning, args.temperature, args.throttle)
        for k, i in enumerate(idxs):
            proba[pos_of[i]] = pr[k]
        print(f"[done] {tg!r}: {len(idxs)} tweets", flush=True)

    pred_ids = proba.argmax(1)
    tgt = np.array([df.at[i, "target"] for i in order_idx], dtype=object)
    out = {"proba": proba, "pred_id": pred_ids, "target": tgt.astype(str), "idx": np.array(order_idx)}
    res = {"model": args.model, "grounded": True, "n": len(order_idx), "runtime_sec": round(time.time() - t0, 1)}
    if labeled:
        y = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in order_idx])
        out["y_true"] = y
        res["pooled_Favg2"] = round(S.compute_metrics(y, pred_ids)["Favg2"] * 100, 2)
        res["per_target"] = {tg: round(S.compute_metrics(y[tgt == tg], pred_ids[tgt == tg])["Favg2"] * 100, 2) for tg in targets}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    json.dump({"cards": cards, "res": res}, open(os.path.splitext(args.out)[0] + "_meta.json", "w"), indent=2, ensure_ascii=False)
    print("[RESULT] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
