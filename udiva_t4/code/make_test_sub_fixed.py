"""Pack the frozen hedge-pack into the Codabench test submission.

The organizer releases the predefined test query as
`evaluation/eval_data/anticipation_template.json` (7 sessions, 651 segments with fixed
segment ids and t_b/t_e). It is filled IN PLACE: every session/segment key and t_b/t_e is
kept untouched, and each participant's entry is replaced by

    {"hypotheses": [{"events": seq_1}, ..., {"events": seq_5}]}

with the five sequences of `heldbest_hedgepack.json` -- the same set for every cell, since
the predictor is input-free. The result is zipped as a single `anticipation.json` at the
zip root. Note that `hypotheses` is the SUBMISSION schema; the development
`starting_kit/anticipation/reference.json` uses the ground-truth shape `"events": seq` and
does not round-trip through the scorer.

Usage: python make_test_sub_fixed.py [--out anticipation_submission.zip]
                                     [--template PATH] [--pack heldbest_hedgepack.json]
"""
import argparse
import hashlib
import json
import os
import zipfile

from udiva import io as IO

HERE = os.path.dirname(os.path.abspath(__file__))
PA, PB = "participant_a", "participant_b"
# sha256 of the anticipation.json inside ranked Codabench submission 825333
SUBMITTED_SHA256 = "ea43f9e899c59c3ab00b1b83461922fb919f6c3dc759079d1d55f41ce21a80a2"


def build(out_path, template=None, pack=None, kmax=5):
    template = template or IO.TEST_TEMPLATE
    pack = pack or os.path.join(HERE, "heldbest_hedgepack.json")
    alts = json.load(open(pack))["alts"]                # up to K=5 event-sequences
    tpl = json.load(open(template))
    for sid, segs in tpl["anticipation"].items():
        for seg_id, seg in segs.items():
            for p in (PA, PB):
                seg["participants"][p] = {
                    "hypotheses": [{"events": [list(e) for e in a]} for a in alts[:kmax]]}
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = json.dumps(tpl)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("anticipation.json", payload)
    nseg = sum(len(v) for v in tpl["anticipation"].values())
    digest = hashlib.sha256(payload.encode()).hexdigest()
    print(f"wrote {out_path} ({len(payload)}B, {nseg} segments, {2 * nseg} cells)")
    print(f"  sha256(anticipation.json) = {digest}")
    print(f"  matches submitted 825333  = {digest == SUBMITTED_SHA256}")
    print("  K=5 hedge-pack alts:", json.dumps(alts))
    s0 = list(tpl["anticipation"])[0]
    g0 = list(tpl["anticipation"][s0])[0]
    print("  sample cell:", json.dumps({s0: {g0: tpl["anticipation"][s0][g0]}})[:400])
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="anticipation_submission.zip")
    ap.add_argument("--template", default=None, help=f"default: {IO.TEST_TEMPLATE}")
    ap.add_argument("--pack", default=None)
    a = ap.parse_args()
    build(a.out, a.template, a.pack)
