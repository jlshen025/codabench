#!/usr/bin/env python3
"""Cheap hand-crafted AUDIO features per GT 2s-segment from the exocentric GF mp4 audio
(16kHz mono). Captures speech activity + action-sound transients (brick clicks etc.):
rms energy, spectral centroid/bandwidth/rolloff, ZCR, onset strength, 5 MFCCs.
Aligned to build_gt seg windows (s_0001..). Saves {segs, feats[n,NF]} per session .npz.
Hand-crafted eventness features first (cheap); wav2vec2 only if this shows signal."""
import os, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import librosa, decord
from udiva import build_all_gt, seg_grid, SESSIONS

GF = "<datasets>/UDIVA-HHOI/development/annotated_sessions/audiovisual/exo/GF"
OUT = os.environ.get("UDIVA_AUDIO_OUT", "<scratch>/t1/audio_feats")
SR = 16000
NAMES = ["rms_m", "rms_s", "cen", "bw", "rolloff", "zcr", "onset_m", "onset_s",
         "mfcc0", "mfcc1", "mfcc2", "mfcc3", "mfcc4"]
NF = len(NAMES)

def seg_feats(a):
    if len(a) < SR // 10:
        return np.zeros(NF, np.float32)
    rms = librosa.feature.rms(y=a)[0]
    cen = librosa.feature.spectral_centroid(y=a, sr=SR)[0]
    bw = librosa.feature.spectral_bandwidth(y=a, sr=SR)[0]
    ro = librosa.feature.spectral_rolloff(y=a, sr=SR)[0]
    zcr = librosa.feature.zero_crossing_rate(a)[0]
    onset = librosa.onset.onset_strength(y=a, sr=SR)
    mfcc = librosa.feature.mfcc(y=a, sr=SR, n_mfcc=5)
    return np.array([rms.mean(), rms.std(), cen.mean(), bw.mean(), ro.mean(), zcr.mean(),
                     onset.mean(), onset.std(), *mfcc.mean(1)], np.float32)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="all")
    ap.add_argument("--gf_dir", default=GF)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--eval", action="store_true", help="eval mode: grid from video duration (no annotations)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); t0 = time.time()
    if args.eval:
        import glob as _glob
        sess = (sorted(os.path.basename(p)[:-4] for p in _glob.glob(f"{args.gf_dir}/*.mp4"))
                if args.sessions == "gfdir" else args.sessions.split(","))
        grids = {sid: seg_grid(len(decord.VideoReader(f"{args.gf_dir}/{sid}.mp4")) /
                               decord.VideoReader(f"{args.gf_dir}/{sid}.mp4").get_avg_fps()) for sid in sess}
    else:
        gt = build_all_gt(); sess = SESSIONS if args.sessions == "all" else args.sessions.split(",")
        grids = {sid: [(sk, tb, te) for sk, (tb, te, s) in gt[sid]["verbal"].items()] for sid in sess}
    done = 0
    for sid in sess:
        op = f"{args.out}/{sid}.npz"
        if os.path.exists(op):
            done += 1; continue
        y, _ = librosa.load(f"{args.gf_dir}/{sid}.mp4", sr=SR, mono=True)
        grid = grids[sid]; segs = [g[0] for g in grid]; feats = []
        for (seg, tb, te) in grid:
            feats.append(seg_feats(y[int(tb * SR):int(te * SR)]))
        np.savez(op, segs=np.array(segs), feats=np.stack(feats))
        done += 1
        print(f"{sid}: {len(segs)} segs -> {op}", flush=True)
    print(f"DONE {done}/{len(sess)} sessions, NF={NF}, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
