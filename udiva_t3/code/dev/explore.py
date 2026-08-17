"""Landscape analysis + naive baselines on the 2 official anticipation sessions."""
import json
from collections import Counter
import _path  # noqa: F401
import udiva_data as U
import sdl

OFFICIAL = ["001080", "181182"]
PA, PB = "participant_a", "participant_b"


def ev_key(e):
    return sdl.ev_key(e)


def train_freqs(exclude):
    """Marginal event-tuple frequencies from raw events of all sessions except `exclude`."""
    cV, cNV, call = Counter(), Counter(), Counter()
    for sid in U.list_sessions():
        if sid in exclude:
            continue
        for a in U.load_raw(sid):
            if a["subject"] not in (PA, PB):
                continue
            e = U.event_tuple(a)
            call[tuple(sdl.ev_key(e))] += 1
            (cV if len(e) == 2 else cNV)[tuple(sdl.ev_key(e))] += 1
    return cV, cNV, call


def key_to_event(k):
    # ev_key -> event list. ('V',u,o) or ('NV',h,l,o); o may be tuple
    if k[0] == "V":
        o = list(k[2]) if isinstance(k[2], tuple) else k[2]
        return [k[1], o]
    o = list(k[3]) if isinstance(k[3], tuple) else k[3]
    return [k[1], k[2], o]


def gt_segments_official():
    out = {}
    ref = json.load(open(U.ANT_REF))["anticipation"]
    for sid in OFFICIAL:
        for seg_id, seg in ref[sid].items():
            out[(sid, seg_id)] = seg
    return out


def observed_prefix_events(sid, t_b, lookback=2.0):
    """Oracle: GT events in (t_b-lookback, t_b], per participant (ordered)."""
    raw = U.load_raw(sid)
    win = [a for a in raw if t_b - lookback < a["start"] <= t_b and a["subject"] in (PA, PB)]
    win.sort(key=lambda a: (a["start"], a["end"]))
    out = {PA: [], PB: []}
    for a in win:
        out[a["subject"]].append(U.event_tuple(a))
    return out


def main():
    gt_segs = gt_segments_official()
    cV, cNV, call = train_freqs(set(OFFICIAL))
    topAll = key_to_event(call.most_common(1)[0][0])
    topNV = [key_to_event(k) for k, _ in cNV.most_common(5)]
    topV = [key_to_event(k) for k, _ in cV.most_common(5)]
    print("most common overall:", topAll)
    print("top NV:", topNV[:3])
    print("top V :", topV[:3])

    # landscape: per-participant-per-segment event counts
    nfull = Counter(); nemptyfull = 0; nV = Counter(); nNV = Counter(); tot = 0
    emptyV = emptyNV = 0
    for key, seg in gt_segs.items():
        g = sdl.ref_seg_to_gt(seg)
        for rho in (PA, PB):
            seq = g[rho]; tot += 1
            nfull[len(seq)] += 1
            v = [e for e in seq if len(e) == 2]; nv = [e for e in seq if len(e) == 3]
            if not seq: nemptyfull += 1
            if not v: emptyV += 1
            if not nv: emptyNV += 1
    print(f"\nlandscape over {tot} (participant,segment) cells:")
    print("  full len dist:", dict(sorted(nfull.items())))
    print(f"  empty full: {nemptyfull}/{tot}={nemptyfull/tot:.2%}  empty verbal: {emptyV/tot:.2%}  empty nonverbal: {emptyNV/tot:.2%}")

    # build predictors -> pred_segments {key:{rho:[alts]}}
    def predict(fn):
        pred = {}
        for key, seg in gt_segs.items():
            sid, seg_id = key
            t_b = seg["t_b"]
            pred[key] = fn(sid, seg, t_b)
            print  # noqa
        return pred

    def pred_const(alts):
        return lambda sid, seg, t_b: {PA: alts, PB: alts}

    def pred_persist(sid, seg, t_b):
        pre = observed_prefix_events(sid, t_b, 2.0)
        return {PA: [pre[PA]], PB: [pre[PB]]}

    def pred_hedge_persist(sid, seg, t_b):
        pre = observed_prefix_events(sid, t_b, 2.0)
        return {rho: [[], pre[rho], [topAll], topNV[:1]] for rho in (PA, PB)}

    baselines = {
        "EMPTY": pred_const([[]]),
        "CONST_top1[all]": pred_const([[topAll]]),
        "CONST_topNV": pred_const([topNV[:1]]),
        "HEDGE5_const": pred_const([[], [topAll], topNV[:1], topV[:1], topNV[:2]]),
        "PERSIST(oracle)": pred_persist,
        "HEDGE_persist(oracle)": pred_hedge_persist,
    }
    print("\n%-24s %7s %7s %7s %7s %7s" % ("baseline", "next", "verbal", "nonverb", "full", "mean4"))
    for name, fn in baselines.items():
        pred = predict(fn)
        r = sdl.score_dataset(gt_segs, pred)
        print("%-24s %7.4f %7.4f %7.4f %7.4f %7.4f" % (
            name, r["next"], r["verbal"], r["nonverbal"], r["full"], r["mean4"]))


if __name__ == "__main__":
    main()
