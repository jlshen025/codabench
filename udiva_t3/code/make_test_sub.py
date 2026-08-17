"""Build the TEST-phase anticipation submission by filling the OFFICIAL released template.

⭐ Root-cause fix (2026-07-03): every earlier TEST submission FAILED for two independent
reasons, both fixed here:
  1. WRONG GRID — the old code fabricated a stride grid (`s_0001`=t_b 4.0, uniform). The
     organizer's predefined test segments live in the RELEASED template
     `evaluation/eval_data/anticipation_template.json` (non-uniform, gappy ids, 651 cells).
     A self-built grid never matches → scorer KeyError → "Child task failed/non-zero rc".
  2. WRONG STRUCTURE — the old code wrote `participants[p]["events"]` (the dev GT shape).
     The submission needs `participants[p]["hypotheses"] = [{"events": seq}, ...]` (up to K=5).
     Both the old `single` and `klist` encodings lacked `hypotheses` → both FAILED identically.

We now fill the template in place: keep every session/segment key + t_b/t_e, replace each
participant's `hypotheses` with the held-best K=5 alt set. Usage: python make_test_sub.py [b2|b4]
"""
import os
import sys
import pack
import predictors as P


def build(which="b2", out_path=None):
    alts = P.alts_for_all(P.b2_factory if which == "b2" else P.b4_factory)
    # Constant (input-free) predictor: the same K=5 alt set for every cell + participant.
    tpl = pack.fill_template(lambda sid, seg, p: alts)
    out = out_path or os.path.join(os.environ.get("UDIVA_T3_OUT", "out"),
                                   f"test_{which}_template.zip")
    p, nbytes, nseg = pack.write_template_zip(tpl, out, json_name="anticipation.json")
    print(f"wrote {p} ({nbytes}B payload, {nseg} cells, {which})")
    print("  K=5 alts:", alts)
    s0 = list(tpl["anticipation"])[0]
    g0 = list(tpl["anticipation"][s0])[0]
    import json
    print("  sample cell:", json.dumps({s0: {g0: tpl["anticipation"][s0][g0]}})[:400])
    return p


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "b2"
    build(which)
