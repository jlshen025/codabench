#!/usr/bin/env python3
"""Build the Track 5 TEST submission zip from a labels file (eid -> letter).

Fills the organizer's MCQ keyset (causal_MCQS.json) with, per effect:
  predicted_option           = the label from the labels file
  predicted_cause_timestamp  = t_b(effect) - k   (constant offset, default k = 1.15 s)
Output: a zip containing a single causal.json at the root, keyed
{"causal": {"<sid>.mp4": {"<eid>": {...}}}} — the accepted submission format.

Usage: python make_submission.py <labels.json> [k] [out.zip]
"""
import json, os, sys, zipfile
from collections import Counter

MCQS = "<datasets>/UDIVA-HHOI/evaluation/eval_data/causal_MCQS.json"


def main():
    labels = json.load(open(sys.argv[1]))
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 1.15
    out_zip = sys.argv[3] if len(sys.argv) > 3 else "causal_submission.zip"
    mcq = json.load(open(MCQS))["SEGMENT"]
    out, n = {"causal": {}}, 0
    for sid_mp4, eids in mcq.items():
        out["causal"][sid_mp4] = {}
        for eid, e in eids.items():
            lab = labels.get(eid)
            assert lab in ("A", "B", "C", "D", "E"), f"missing/invalid label for {eid}"
            out["causal"][sid_mp4][eid] = {
                "predicted_cause_timestamp": round(float(e["effect"]["t_b"]) - k, 3),
                "predicted_option": lab,
            }
            n += 1
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("causal.json", json.dumps(out, ensure_ascii=False))
    print(f"wrote {out_zip}: {n} effects | k={k} | "
          f"label dist {dict(Counter(l['predicted_option'] for s in out['causal'].values() for l in s.values()))}")


if __name__ == "__main__":
    main()
