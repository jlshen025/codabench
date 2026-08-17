#!/usr/bin/env python3
"""Parse .srt transcripts -> utterances; build per-segment per-speaker text."""
import os, re

DATA = "<datasets>/UDIVA-HHOI/development/annotated_sessions"
TRANS = f"{DATA}/transcripts"
SPK = {"PART.1": "participant_a", "PART.2": "participant_b", "SUPERVISOR": "supervisor"}

def _t(s):
    h, m, rest = s.split(":"); sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0

def parse_srt(path):
    out = []
    blocks = re.split(r"\n\s*\n", open(path, encoding="utf-8", errors="replace").read())
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        if len(lines) < 2: continue
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None: continue
        a, c = lines[ti].split("-->")
        try: start, end = _t(a.strip()), _t(c.strip())
        except Exception: continue
        text = " ".join(lines[ti + 1:]).strip()
        m = re.match(r"^(PART\.1|PART\.2|SUPERVISOR)\s*:\s*(.*)$", text)
        if m:
            spk = SPK[m.group(1)]; txt = m.group(2).strip()
        else:
            spk = "unknown"; txt = text
        out.append({"start": start, "end": end, "speaker": spk, "text": txt})
    return out

EVAL_TRANS = "<datasets>/UDIVA-HHOI/evaluation/transcripts"

def srt_path(sid):
    p = f"{TRANS}/{sid}.srt"
    return p if os.path.exists(p) else f"{EVAL_TRANS}/{sid}.srt"  # eval sids (disjoint) fall back to eval dir

def seg_speaker_text(sid, grid, path=None):
    """grid: list/iterable of (seg_key, t_b, t_e). Returns {seg: {speaker: text}} for
    participant_a/b utterances overlapping the segment."""
    utts = parse_srt(path or srt_path(sid))
    out = {}
    for sk, tb, te in grid:
        d = {}
        for u in utts:
            if u["speaker"] in ("participant_a", "participant_b") and u["start"] < te and u["end"] > tb:
                d.setdefault(u["speaker"], []).append(u["text"])
        out[sk] = {sp: " ".join(t) for sp, t in d.items()}
    return out

if __name__ == "__main__":
    from udiva import build_gt, SESSIONS
    tot = 0
    for sid in SESSIONS:
        u = parse_srt(srt_path(sid)); tot += len(u)
    print("total utterances across 21 sessions:", tot)
    sid = "001080"
    gt = build_gt(sid)
    grid = [(sk, tb, te) for sk, (tb, te, s) in gt["verbal"].items()]
    st = seg_speaker_text(sid, grid)
    for sk in list(st)[:4]:
        print(sk, st[sk])
