"""
llm_grounded2.py — grounded voter v2: AIM-DIRECTED structured reasoning (LC-CoT-lite).

Residual finding (2026-07-02, Digital LOTO, held-best blend): F_against=69.5 driven by
PRECISION (94 false-Against vs 28 missed-Against) — voters misattribute criticism of the
OLD SYSTEM / poor implementation / obstacles as Against-the-target, and misapply the
sarcasm→Against heuristic. Fix = make the model resolve the evaluative cue's AIM before
labeling (T=target / O=other:old-system,implementation,obstacle,opponent / U=unrelated),
with the opposite-polarity mapping for O (criticizing the old way or the target's
opponents = Favor). Target-agnostic METHOD (no per-target tuning) → Gate-2 safe.

Same card + API + npz conventions as llm_grounded.py; --cards_from reuses a prior run's
cards so prompt-v2 is the ONLY variable. --target_filter + --limit allow a cheap paired
probe on one target slice before full runs. HTTP path (llm_predict.call_api).
"""
import os, sys, json, argparse, time, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from llm_predict import _provider_for, call_api
from llm_grounded import make_card

# label may sit in field 3 (aim format) — accept both '<n>|<T/O/U>|<Label>' and '<n>|<Label>'
LBL2_RE = re.compile(r'^\s*(\d+)\s*\|\s*(?:([TOU])\s*\|\s*)?(favor|against|none)\b', re.I)


def sys_prompt_v2(target, card, variant="v2"):
    lines = ["You are an expert annotator for Arabic stance detection. Classify the AUTHOR's stance "
             "TOWARD THE TARGET as exactly one of: Favor, Against, None.",
             f"TARGET: {target}"]
    if card.get("definition"):
        lines.append(f"MEANING: {card['definition']}")
    if card.get("aspects"):
        lines.append(f"RELATED ASPECTS: {card['aspects']}")
    common_head = [
        "For EACH tweet, first find the evaluative CUE (praise, criticism, sarcasm, demand, fear, hope),",
        "then decide the cue's AIM before labeling:",
        "- T: aimed at the TARGET itself (its existence, value, adoption, mandate) or one of its aspects.",
        "- O: aimed at something ELSE — the OLD/pre-target way of doing things, a poor IMPLEMENTATION or",
        "  outage of it, an obstacle to it, or people who OPPOSE the target.",
        "- U: no evaluative content about the target (unrelated, purely factual/news).",
        "Mapping to the label:",
        "- T: positive cue -> Favor; negative/sarcastic cue -> Against.",
    ]
    if variant == "v2":
        o_rule = [
            "- O: usually the OPPOSITE polarity toward the target: criticizing the OLD system, demanding faster",
            "  adoption, or attacking the target's opponents = Favor. A complaint about an implementation glitch",
            "  or outage alone does NOT mean Against the target's idea — label Against ONLY if the author",
            "  rejects the target itself; if they still endorse the goal, label Favor.",
        ]
    else:  # v2b — asymmetric default: flip O to Favor ONLY on an explicit pro-goal signal
        o_rule = [
            "- O: decide by what the cue implies ABOUT THE TARGET, with a conservative default:",
            "  * Criticizing the OLD/pre-target way, DEMANDING faster/wider adoption, or attacking people who",
            "    oppose the target = Favor (these are explicit pro-target signals).",
            "  * A complaint about the target's implementation, services, outages, or results = Against by",
            "    default — flip to Favor ONLY if the SAME tweet also carries an explicit pro-target signal",
            "    (praise of the concept, a wish for it to succeed/expand, criticism of the old way).",
        ]
    tail = [
        "- U: None. Do NOT default to None merely because the target is not named literally; indirect stance",
        "  via an aspect still counts (that is aim T).",
        "Sarcasm counts toward whatever it is AIMED at: sarcasm about the target = Against; sarcasm about",
        "the old system or the target's critics = Favor.",
        "Output EXACTLY one line per tweet formatted '<number>|<T/O/U>|<Label>' (Label = Favor, Against,",
        "or None). No other text.",
    ]
    return "\n".join(lines + common_head + o_rule + tail)


def classify_target(texts, cfg, model, sysp, batch, reasoning):
    proba = np.zeros((len(texts), 3), dtype=float)
    aims = [""] * len(texts)
    for b0 in range(0, len(texts), batch):
        chunk = texts[b0:b0 + batch]
        user = "Classify each tweet's stance toward the TARGET:\n\n" + "\n".join(
            f"{i+1}| {t}" for i, t in enumerate(chunk))
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": user}]
        content = call_api(cfg, model, msgs, reasoning, 120 * len(chunk) + 1000, 0.0)
        got, gaim = {}, {}
        for line in content.splitlines():
            mm = LBL2_RE.match(line)
            if mm:
                got[int(mm.group(1))] = mm.group(3).capitalize()
                gaim[int(mm.group(1))] = (mm.group(2) or "").upper()
        miss = [k for k in range(1, len(chunk) + 1) if k not in got]
        if len(miss) > max(2, len(chunk) // 5):
            content = call_api(cfg, model, msgs, reasoning, 120 * len(chunk) + 1000, 0.0)
            for line in content.splitlines():
                mm = LBL2_RE.match(line)
                if mm:
                    got.setdefault(int(mm.group(1)), mm.group(3).capitalize())
                    gaim.setdefault(int(mm.group(1)), (mm.group(2) or "").upper())
        for j in range(len(chunk)):
            proba[b0 + j, S.LABEL2ID.get(got.get(j + 1, "None"), 2)] += 1.0
            aims[b0 + j] = gaim.get(j + 1, "")
        print(f"  batch {b0//batch+1}/{(len(texts)+batch-1)//batch}", flush=True)
    return proba, aims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4.5")
    ap.add_argument("--csv", default=S.DEV_CSV)
    ap.add_argument("--order", default="file", choices=["loto", "file"])
    ap.add_argument("--target_filter", default="", help="only rows of this target (probe)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows AFTER filter (probe); 0=all")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--variant", default="v2", choices=["v2", "v2b"])
    ap.add_argument("--cards_from", default="", help="reuse cards.json from a prior run's meta")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pname, cfg = _provider_for(args.model); wire = cfg["wire_model"]
    print(f"[provider] {pname} wire={wire}", flush=True)

    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    labeled = "stance" in df.columns and (df["stance"].str.strip() != "").all()
    if args.order == "loto":
        order_idx = []
        for tg in dict.fromkeys(df["target"].tolist()):
            order_idx.extend(df.index[df["target"] == tg].tolist())
    else:
        order_idx = df.index.tolist()
    if args.target_filter:
        order_idx = [i for i in order_idx if df.at[i, "target"] == args.target_filter]
    if args.limit:
        order_idx = order_idx[: args.limit]
    targets = list(dict.fromkeys(df.loc[order_idx, "target"].tolist()))

    cards = {}
    if args.cards_from:
        cards = dict(json.load(open(args.cards_from))["cards"])
        print(f"[cards] reusing {list(cards)}", flush=True)
    t0 = time.time()
    for tg in targets:
        if tg not in cards:
            cards[tg] = make_card(cfg, wire, tg)
        print(f"[card] {tg!r}: def={cards[tg]['definition'][:50]!r}", flush=True)

    proba = np.zeros((len(order_idx), 3), dtype=float)
    aims = np.array([""] * len(order_idx), dtype=object)
    pos_of = {gi: k for k, gi in enumerate(order_idx)}
    for tg in targets:
        idxs = [i for i in order_idx if df.at[i, "target"] == tg]
        texts = [df.at[i, "text"] for i in idxs]
        pr, am = classify_target(texts, cfg, wire, sys_prompt_v2(tg, cards[tg], args.variant), args.batch, args.reasoning)
        for k, i in enumerate(idxs):
            proba[pos_of[i]] = pr[k]; aims[pos_of[i]] = am[k]
        print(f"[done] {tg!r}: {len(idxs)} tweets", flush=True)

    pred_ids = proba.argmax(1)
    tgt = np.array([df.at[i, "target"] for i in order_idx], dtype=object)
    out = {"proba": proba, "pred_id": pred_ids, "target": tgt.astype(str),
           "idx": np.array(order_idx), "aim": aims.astype(str)}
    res = {"model": args.model, "grounded": f"{args.variant}-aim", "n": len(order_idx),
           "filter": args.target_filter, "limit": args.limit,
           "runtime_sec": round(time.time() - t0, 1)}
    if labeled:
        y = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in order_idx])
        out["y_true"] = y
        res["pooled_Favg2"] = round(S.compute_metrics(y, pred_ids)["Favg2"] * 100, 2)
        res["per_target"] = {tg: round(S.compute_metrics(y[tgt == tg], pred_ids[tgt == tg])["Favg2"] * 100, 2) for tg in targets}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    json.dump({"cards": cards, "res": res}, open(os.path.splitext(args.out)[0] + "_meta.json", "w"),
              indent=2, ensure_ascii=False)
    print("[RESULT] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
