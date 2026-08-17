"""Dataset access: session list, GT, transcripts, metadata, video paths."""
import os, re, json, csv
from functools import lru_cache
from .parse import parse_session, load_raw

ROOT = "<datasets>/UDIVA-HHOI/development"
ANN_DIR = f"{ROOT}/annotated_sessions/annotations"
TRANS_DIR = f"{ROOT}/annotated_sessions/transcripts"
META_DIR = f"{ROOT}/annotated_sessions/metadata"
AV = f"{ROOT}/annotated_sessions/audiovisual"
UNANN_AV = f"{ROOT}/unannotated_sessions/audiovisual"
UNANN_TRANS = f"{ROOT}/unannotated_sessions/transcripts"
REF_JSON = f"{ROOT}/annotated_sessions/starting_kit/recognition/reference.json"

# TEST/eval split (decrypted; GT withheld). 7 segment sessions, E1/E2/GF mp4 + .srt.
EVAL_ROOT = "<datasets>/UDIVA-HHOI/evaluation"
EVAL_AV = f"{EVAL_ROOT}/audiovisual"
EVAL_TRANS = f"{EVAL_ROOT}/transcripts"


def all_sids():
    return sorted(s[:-5] for s in os.listdir(ANN_DIR) if s.endswith(".json"))


def eval_sids():
    """The 7 TEST/eval session ids (from the eval E1 mp4s)."""
    return sorted(s[:-4] for s in os.listdir(f"{EVAL_AV}/ego/E1") if s.endswith(".mp4"))


@lru_cache(maxsize=256)
def gt_session(sid, n_segments=None):
    # Memoized: annotation parse is expensive and re-requested per CV fold. All callers
    # (load_gt / compose_nonverbal / metric / cv) treat the result read-only.
    return parse_session(load_raw(ANN_DIR, sid), n_segments=n_segments)


def load_gt(sids):
    gt = {"verbal": {}, "nonverbal": {}}
    for sid in sids:
        p = gt_session(sid)
        for st in ("verbal", "nonverbal"):
            gt[st][sid] = p[st]
    return gt


def n_segments(sid):
    """Number of 2s segments derived from GT (max event end)."""
    return len(gt_session(sid)["verbal"])


_SRT_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def _to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def load_transcript(sid, trans_dir=TRANS_DIR):
    """Return list of cues: {start,end,speaker in {participant_a,participant_b,supervisor,none}, text}."""
    path = os.path.join(trans_dir, f"{sid}.srt")
    if not os.path.exists(path):
        return []
    blocks = [b for b in open(path, encoding="utf-8", errors="replace").read().strip().split("\n\n") if b.strip()]
    cues = []
    for b in blocks:
        lines = b.splitlines()
        ts = None
        text_lines = []
        for ln in lines:
            m = _SRT_TS.search(ln)
            if m:
                ts = (_to_sec(*m.groups()[:4]), _to_sec(*m.groups()[4:]))
            elif ln.strip().isdigit() and ts is None and not text_lines:
                continue
            else:
                text_lines.append(ln)
        if ts is None:
            continue
        text = " ".join(text_lines).strip()
        spk = "none"
        mm = re.match(r"\s*(PART\.1|PART\.2|SUPERVISOR)\s*:\s*(.*)", text, re.I | re.S)
        if mm:
            tag = mm.group(1).upper()
            spk = {"PART.1": "participant_a", "PART.2": "participant_b", "SUPERVISOR": "supervisor"}[tag]
            text = mm.group(2).strip()
        cues.append({"start": ts[0], "end": ts[1], "speaker": spk, "text": text})
    return cues


def session_meta():
    """sid -> dict(p1,p2,language,lego,difficulty,known,...)."""
    out = {}
    f = os.path.join(META_DIR, "sessions_spotting.csv")
    for row in csv.DictReader(open(f)):
        sid = row["ID"].zfill(6)
        out[sid] = {"p1": row["PART.1"], "p2": row["PART.2"], "language": row.get("LANGUAGE", ""),
                    "lego": row.get("LEGO", ""), "which_lego": row.get("WHICH_LEGO", ""),
                    "known": row.get("KNOWN", ""), "lego_order": row.get("LEGO_ORDER", "")}
    return out


def video_path(sid, view, annotated=True, split=None):
    """view in {E1,E2,GF}. Returns mp4 path. split='eval' → the TEST/eval videos."""
    if split == "eval":
        base = EVAL_AV
    else:
        base = AV if annotated else UNANN_AV
    sub = {"E1": "ego/E1", "E2": "ego/E2", "GF": "exo/GF"}[view]
    return os.path.join(base, sub, f"{sid}.mp4")
