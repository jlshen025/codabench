"""Package a prediction dataset into a Codabench result zip for Track 4.

Submission JSON mirrors reference.json: {"anticipation": {sess: {seg: {t_b,t_e,
participants:{participant_a/b:{events: <seq>}}}}}}. `events` is FLAT (list of event
tuples) = the K=1 form (byte-identical to reference.json), or a nested list of up to
5 alternative sequences for the K=5 form.
"""
import json
import os
import zipfile


def to_submission(pred):
    """Wrap a parsed pred dataset under the 'anticipation' root key, keeping only the
    fields the scorer needs. pred[sess][seg]['participants'][p]['events'] = seq or alts."""
    out = {}
    for sess, segs in pred.items():
        out[sess] = {}
        for seg, sd in segs.items():
            out[sess][seg] = {
                "t_b": sd["t_b"], "t_e": sd["t_e"],
                "participants": {
                    p: {"events": sd["participants"][p]["events"]}
                    for p in sd["participants"]
                },
            }
    return {"anticipation": out}


def write_zip(pred, zip_path, arcname="anticipation.json"):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    payload = to_submission(pred)
    tmp_json = zip_path + ".json"
    json.dump(payload, open(tmp_json, "w"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tmp_json, arcname)
    os.remove(tmp_json)
    return zip_path
