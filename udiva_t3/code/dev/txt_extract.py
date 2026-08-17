"""Transcript-text scout: encode the recent conversation (.srt prefix up to t_b) with a
multilingual sentence-transformer (LOCAL model, Gate-1 OK) into the SAME feature format as
the video features, so vidprobe's k-NN can test it. If text features don't beat the const
prior (mean4 0.4127), the transcript-NLU family is ruled out.
"""
import os, sys, re
os.environ["HF_HOME"] = "<cache>/huggingface"
import numpy as np
import _path  # noqa: F401
import udiva_data as U
import cv as CV

TXT_DIR = U.WORK_DIR / "txtfeats"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
W = 10.0   # recent-conversation lookback seconds
TS_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")


def parse_srt(sid):
    raw = open(U.TRANS_DIR / f"{sid}.srt", encoding="utf-8", errors="ignore").read()
    utts = []
    for blk in re.split(r"\n\s*\n", raw.strip()):
        lines = [l for l in blk.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        m = TS_RE.search(lines[1])
        if not m:
            continue
        h, mi, s, ms = map(int, m.group(1, 2, 3, 4))
        start = h * 3600 + mi * 60 + s + ms / 1000.0
        text = " ".join(lines[2:]).strip()
        utts.append((start, text))
    return utts


def recent_text(utts, t_b):
    rec = [t for (st, t) in utts if t_b - W < st < t_b]
    return " ".join(rec) if rec else ""


def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    gt = CV.build_all_gt()
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    for sid in U.list_sessions():
        utts = parse_srt(sid)
        segs = list(gt[sid].items())
        texts = [recent_text(utts, seg["t_b"]) for _, seg in segs]
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True).astype(np.float32)  # [n,D]
        feat = emb[:, None, :]  # [n,1,D]
        np.savez_compressed(
            TXT_DIR / f"{sid}.npz",
            seg_ids=np.array([s for s, _ in segs]),
            t_b=np.array([seg["t_b"] for _, seg in segs], dtype=np.float32),
            featA=feat, featB=feat)
        ne = sum(1 for t in texts if t)
        print(f"  {sid}: {len(segs)} segs, {ne} with text, D={emb.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
