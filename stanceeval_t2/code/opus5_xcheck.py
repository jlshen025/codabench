"""opus5_xcheck.py — prior-calibrated re-read of opus-flagged rows by a
non-claude reader (sol / deepseek), one pool at a time.

Same instrument class as none_recheck.py (measured-effective), generalized:
--pool selects the pool-specific calibrated prompt; --rows gives the row ids.
Aggregate priors only — no per-row gold known or used. Gate-1 clean.
Run: <venv-python> opus5_xcheck.py --pool ecF ...
"""
import os, sys, json, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_predict import _provider_for, call_api

COMMON = (
    "You are an expert annotator for Arabic stance detection on the target: {tname}.\n"
    "The corpus annotators are LIBERAL about assigning a stance: implicit or indirect "
    "evaluative lean counts as a stance; None is reserved for tweets with no evaluative "
    "colour toward the target at all (pure news relays, logistics, off-topic).\n"
    "{cal}\n"
    "Judge each tweet independently on its text (AR authoritative, EN is machine translation).\n"
    "Output EXACTLY one line per tweet: '<number>|<Label>|<confidence 0-100>' "
    "(Label = Favor, Against, or None). No other text."
)

TNAME = {"ec": "electric cars (السيارات الكهربائية)",
         "tr": "the three-term school year system (نظام الفصول الدراسية الثلاثة)"}

CAL = {
    "ecN": "CALIBRATION: a previous pass labelled every tweet below 'None'. Measurement shows "
           "a handful (roughly 1 in 6 of THIS shortlist) are actually FAVOR or AGAINST. Find them; "
           "keep None where there is genuinely no stance.",
    "trN": "CALIBRATION: a previous pass labelled every tweet below 'None'. Measurement shows "
           "about 2 of the full 14-row pool are actually AGAINST (none are Favor). Be precise.",
    "ecA": "CALIBRATION: a previous pass labelled every tweet below 'Against'. Measurement shows "
           "a few of the full 125-row pool are actually FAVOR (e.g. desire to buy despite complaints, "
           "rebutting critics) and a few are None (no stance on EVs themselves). Decide each afresh.",
    "trA": "CALIBRATION: a previous pass labelled every tweet below 'Against'. Measurement shows "
           "about 3 of the full 217-row pool are actually FAVOR (defending the system or mocking its "
           "critics) and about 6 are None (pure news/off-topic). Decide each afresh.",
    "ecF": "CALIBRATION: a previous pass labelled every tweet below 'Favor'. Measurement shows "
           "about 1 in 10 of the full 134-row pool is actually NONE — pure news/product reports with "
           "no author evaluation. Genuine enthusiasm, desire or positive framing stays Favor.",
    "trF": "CALIBRATION: a previous pass labelled every tweet below 'Favor'. Measurement shows "
           "about 2 of the full 70-row pool are actually AGAINST (sarcastic fake praise). Be precise.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--provider", default="sub2apigpt")
    ap.add_argument("--pool", required=True, choices=list(CAL))
    ap.add_argument("--rows", required=True, help="comma-separated row ids")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _, cfg = _provider_for(args.model, args.provider); wire = cfg["wire_model"]
    idx = [int(x) for x in args.rows.split(",") if x.strip()]
    sysmsg = COMMON.format(tname=TNAME[args.pool[:2]], cal=CAL[args.pool])
    ar = pd.read_csv("test_norm.csv", keep_default_na=False, dtype=str)
    en = pd.read_csv("_llm/test_en.csv", keep_default_na=False, dtype=str)
    res = {}
    for b0 in range(0, len(idx), args.batch):
        chunk = idx[b0:b0 + args.batch]
        lines = []
        for k, i in enumerate(chunk):
            a = " ".join(ar.at[int(i), "text"].split())
            e = " ".join(en.at[int(i), "text"].split())
            lines.append(f"{k+1}| AR: {a}\n   EN: {e}")
        content = call_api(cfg, wire, [{"role": "system", "content": sysmsg},
                                       {"role": "user", "content": "\n".join(lines)}],
                           "none", 60 * len(chunk) + 600, 0.0)
        for ln in content.splitlines():
            p = [x.strip() for x in ln.split("|")]
            if len(p) >= 2 and p[0].isdigit():
                k = int(p[0]) - 1
                if 0 <= k < len(chunk):
                    lab = p[1].capitalize()
                    conf = int(p[2]) if len(p) > 2 and p[2].isdigit() else 50
                    if lab in ("Favor", "Against", "None"):
                        res[int(chunk[k])] = {"label": lab, "conf": conf}
        print(f"  {min(b0+args.batch, len(idx))}/{len(idx)}", flush=True)
    json.dump(res, open(args.out, "w"), indent=0)
    print(f"[{args.model} {args.pool}] {len(res)}/{len(idx)} judged -> {args.out}")
    for i in idx:
        if i in res:
            print(f"  {i}: {res[i]['label']}({res[i]['conf']})")


if __name__ == "__main__":
    main()
