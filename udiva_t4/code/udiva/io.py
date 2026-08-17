"""I/O + parsing for UDIVA-HHOI Track 4.

Parser VALIDATED against starting_kit/anticipation/reference.json (001080,181182):
253/254 (participant,segment) cells exact. Rule:
- window membership: t_b < event.start <= t_e   (exclusive low, inclusive high)
- order within segment: sort by (start, end)
- event -> tuple: V -> [utterance_type, target]; NV -> [high, low, target]
  where target = target_filtered collapsed (single-element list -> the string).
- only subjects participant_a / participant_b.
"""
import json
import os
import glob

# Point UDIVA_ROOT at the unpacked challenge data (the directory holding
# `development/` and `evaluation/`); everything else is derived from it.
UDIVA_ROOT = os.environ.get("UDIVA_ROOT", "<datasets>/UDIVA-HHOI")
DATA_ROOT = os.path.join(UDIVA_ROOT, "development")
ANN_DIR = os.path.join(DATA_ROOT, "annotated_sessions", "annotations")
TRANS_DIR = os.path.join(DATA_ROOT, "annotated_sessions", "transcripts")
REF_PATH = os.path.join(DATA_ROOT, "annotated_sessions", "starting_kit",
                        "anticipation", "reference.json")
TEST_TEMPLATE = os.path.join(UDIVA_ROOT, "evaluation", "eval_data",
                             "anticipation_template.json")
HORIZON = 2.0


def list_sessions():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(ANN_DIR, "*.json")))


def load_annotations(sess):
    return json.load(open(os.path.join(ANN_DIR, f"{sess}.json")))["annotations"]


def load_dev_reference():
    """The 2-session organizer GT (001080, 181182) = the DEV eval set."""
    return json.load(open(REF_PATH))["anticipation"]


def _collapse_target(tf):
    if isinstance(tf, list):
        return tf[0] if len(tf) == 1 else list(tf)
    return tf


def event_to_tuple(e):
    t = _collapse_target(e["target_filtered"])
    if e["act"] == "V":
        return [e["utterance_type"], t]
    return [e["high_level_action"], e["low_level_action"], t]


def parse_segment(ann, t_b, t_e):
    """Events occurring in horizon (t_b, t_e], per participant, ordered."""
    ev = {"participant_a": [], "participant_b": []}
    win = [e for e in ann if t_b < e["start"] <= t_e]
    win.sort(key=lambda e: (e["start"], e["end"]))
    for e in win:
        if e["subject"] in ev:
            ev[e["subject"]].append(event_to_tuple(e))
    return ev


def make_segment(ann, t_b):
    t_e = t_b + HORIZON
    ev = parse_segment(ann, t_b, t_e)
    return {"t_b": t_b, "t_e": t_e,
            "participants": {p: {"events": ev[p]} for p in ("participant_a", "participant_b")}}


def parse_session_with_tbs(sess, t_bs):
    ann = load_annotations(sess)
    return {f"s_{i+1:04d}": make_segment(ann, tb) for i, tb in enumerate(t_bs)}


def session_duration(ann):
    return max((e["end"] for e in ann), default=0.0)


def grid_tbs(ann, stride=2.0, start=0.0, warmup=0.0):
    """Fixed-stride reference timestamps tiling a session (for CV)."""
    dur = session_duration(ann)
    tbs, t = [], start + warmup
    while t + HORIZON <= dur + 1e-6:
        tbs.append(round(t, 3))
        t += stride
    return tbs


def build_cv_dataset(sessions, stride=2.0):
    """Parse each session into anticipation segments on a fixed-stride grid."""
    ds = {}
    for s in sessions:
        ann = load_annotations(s)
        tbs = grid_tbs(ann, stride=stride)
        ds[s] = {f"s_{i+1:04d}": make_segment(ann, tb) for i, tb in enumerate(tbs)}
    return ds


# ---------------- transcripts (.srt) ----------------
import re

_SPK = {"PART.1": "participant_a", "PART.2": "participant_b", "SUPERVISOR": "supervisor"}
_TC = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)")


def parse_srt(path):
    """Return [(start, end, participant, text)], times in seconds, speaker mapped to
    participant_a/participant_b/supervisor."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for b in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in b.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        ti = next((i for i, l in enumerate(lines) if _TC.match(l)), None)
        if ti is None:
            continue
        m = _TC.match(lines[ti])
        def sec(h, mn, s, ms):
            return int(h) * 3600 + int(mn) * 60 + int(s) + int(ms) / 1000.0
        st = sec(*m.group(1, 2, 3, 4)); en = sec(*m.group(5, 6, 7, 8))
        text = " ".join(lines[ti + 1:]).strip()
        sp = re.match(r"\s*(PART\.[12]|SUPERVISOR)\s*:\s*(.*)", text)
        spk = _SPK.get(sp.group(1)) if sp else None
        body = sp.group(2) if sp else text
        out.append((st, en, spk, body))
    return out


def load_transcript(sess):
    return parse_srt(os.path.join(TRANS_DIR, f"{sess}.srt"))


def observed_utterances(utts, t_b, lookback=None):
    """Utterances fully observed by t_b (start < t_b). Optionally only the last `lookback` s."""
    obs = [u for u in utts if u[0] < t_b]
    if lookback is not None:
        obs = [u for u in obs if u[0] >= t_b - lookback]
    return obs


# ---------------- event-anchored grid (FAITHFUL eval simulation) ----------------
def event_anchored_tbs(ann, delta=0.25, min_gap=2.0):
    """Reference timestamps anchored just before participant event onsets (the eval
    samples t_b ~0.3s before an onset). t_b = onset - delta, deduped to min_gap spacing."""
    starts = sorted(set(round(e["start"], 3) for e in ann
                        if e["subject"] in ("participant_a", "participant_b")))
    tbs, last = [], -1e9
    for st in starts:
        tb = round(st - delta, 3)
        if tb <= 0:
            continue
        if tb - last >= min_gap:
            tbs.append(tb); last = tb
    return tbs


def build_cv_dataset_anchored(sessions, delta=0.25, min_gap=2.0):
    ds = {}
    for s in sessions:
        ann = load_annotations(s)
        tbs = event_anchored_tbs(ann, delta, min_gap)
        ds[s] = {f"s_{i+1:04d}": make_segment(ann, tb) for i, tb in enumerate(tbs)}
    return ds
