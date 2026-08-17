"""Build a Track-3 anticipation submission zip.

⭐ CORRECT SUBMISSION FORMAT (verified 2026-07-03 against the RELEASED test query):
Fill the official template `evaluation/eval_data/anticipation_template.json` IN PLACE — it
carries the exact predefined test segment grid (session ids, segment ids, t_b/t_e). Per
participant the submission uses::

    "participants": {"participant_a": {"hypotheses": [{"events": seq1}, {"events": seq2}, ...]}}

i.e. a `hypotheses` LIST of up to K=5 objects, each `{"events": <event-sequence>}`.
Use `fill_template()` for this. See `pack_fixed`/`make_test_sub.py`.

⚠️ Do NOT hand-build a grid and do NOT use `participants[p]["events"]` directly:
  - the dev `reference.json` (GT) uses `"events": seq` — that is the GROUND-TRUTH shape, NOT
    the submission shape. Mirroring it (as the old `build_submission_json` below did) is why
    every 06/07-01..02 submission FAILED ("Child task failed/non-zero rc").
  - a self-built stride grid never matches the organizer's predefined segment ids → KeyError.
The legacy `build_submission_json`/`write_zip` helpers are kept only for the old dev probes.
"""
import json
import os
import zipfile
import udiva_data as U

PA, PB = "participant_a", "participant_b"

# The RELEASED test query grid + the canonical submission schema to fill
# (path configured via UDIVA_HHOI_ROOT — see udiva_data.py).
ANT_TEMPLATE = U.ANT_TEMPLATE


def fill_template(alts_fn, template_path=ANT_TEMPLATE, kmax=5):
    """Fill the official anticipation template with per-cell K-alt hypotheses.

    alts_fn(sid, seg, participant) -> list of event-sequences (each seq = list of event
    tuples; a verbal tuple is [u,o] len-2, a non-verbal tuple is [h,l,o] len-3). Returns the
    full template dict, ready for `write_template_zip`. Keeps every session/segment key and
    the template's own t_b/t_e untouched (they define the exact cells the scorer looks up).
    """
    tpl = json.load(open(template_path))
    for sid, segs in tpl["anticipation"].items():
        for seg_id, seg in segs.items():
            for p in (PA, PB):
                alts = alts_fn(sid, seg, p) or [[]]
                seg["participants"][p] = {
                    "hypotheses": [{"events": [list(e) for e in a]} for a in alts[:kmax]]}
    return tpl


def write_template_zip(tpl, zip_path, json_name="anticipation.json"):
    """Write the filled template as a single-file zip (json at the ROOT)."""
    if os.path.dirname(zip_path):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    payload = json.dumps(tpl)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(json_name, payload)
    nseg = sum(len(v) for v in tpl["anticipation"].values())
    return zip_path, len(payload), nseg


def build_submission_json(grid, pred_fn, encoding="single", kmax=5):
    """grid: {sid: {seg_id: ref-style seg (provides t_b/t_e)}}."""
    out = {"anticipation": {}}
    for sid, segs in grid.items():
        out["anticipation"][sid] = {}
        for seg_id, seg in segs.items():
            pr = pred_fn(sid, seg)
            d = {"t_b": seg["t_b"], "t_e": seg["t_e"], "participants": {}}
            for p in (PA, PB):
                alts = pr.get(p) or [[]]
                if encoding == "single":
                    d["participants"][p] = {"events": list(alts[0])}
                else:
                    d["participants"][p] = {"events": [list(a) for a in alts[:kmax]]}
            out["anticipation"][sid][seg_id] = d
    return out


def write_zip(json_obj, zip_path, json_name="anticipation.json"):
    if os.path.dirname(zip_path):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    payload = json.dumps(json_obj)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(json_name, payload)
    return zip_path, len(payload)


def official_grid_gt():
    """{sid: {seg_id: ref seg}} for the 2 official anticipation sessions."""
    ref = json.load(open(U.ANT_REF))["anticipation"]
    return {sid: dict(segs) for sid, segs in ref.items()}


def perfect_pred_fn(sid, seg):
    """Return the GT events themselves as the single alt (perfect prediction)."""
    pp = seg["participants"]
    return {PA: [pp[PA]["events"]], PB: [pp[PB]["events"]]}


def empty_pred_fn(sid, seg):
    return {PA: [[]], PB: [[]]}


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "perfect"
    name = sys.argv[2] if len(sys.argv) > 2 else "anticipation.json"
    grid = official_grid_gt()
    fn = {"perfect": perfect_pred_fn, "empty": empty_pred_fn}[kind]
    obj = build_submission_json(grid, fn, encoding="single")
    out = os.path.join(os.environ.get("UDIVA_T3_OUT", "out"), f"{kind}_official.zip")
    p, n = write_zip(obj, out, json_name=name)
    # sanity: count segments + sample
    nseg = sum(len(v) for v in obj["anticipation"].values())
    print(f"wrote {p}  ({n} bytes payload, {nseg} segments, json_name={name})")
    sid0 = list(obj["anticipation"])[0]
    seg0 = list(obj["anticipation"][sid0])[0]
    print("sample:", json.dumps({sid0: {seg0: obj["anticipation"][sid0][seg0]}})[:400])
