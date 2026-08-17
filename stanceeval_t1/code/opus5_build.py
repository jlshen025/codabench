#!/usr/bin/env python
"""Assemble the opus-5 WD grounded voter and build the opus5-upgraded cl6v/N14
candidate (opus-5 replaces claude-opus-4-8 in the winning 6-voter ensemble).

Architecture reproduced exactly: softavg(luna,g55,lunaFS,g55FS,<claude-opus>,son5)
+ 1e-3*ce tiebreak -> argmax, then None operating-point reduced to N14 (keep the
14 highest None-margin rows as None, flip the rest to their non-None argmax).

Prints diligence: opus5 distribution vs the true prior A160/F158/N34, decorrelation
vs every incumbent voter, and the row-level diff of the opus5 candidate vs the
locked champion (0.888076).  No server call; selection is offline judgment.
"""
import sys, zipfile, collections
import numpy as np
import stance_lib as S

F = "_eval"; R = "results/eval_deploy_all/_llm"
ID2 = {0: "Against", 1: "Favor", 2: "None"}
ce = (0.15 * np.load(f"{R}/enc_ce.npz")["proba"] + 0.85 * np.load(f"{R}/dep_qens.npz")["proba"]).astype(float)
BS = 32
N = 352


def dist(a):
    c = collections.Counter(a.tolist())
    return {ID2[k]: int(c.get(k, 0)) for k in range(3)}


def assemble_opus5():
    import re, os
    LBL = re.compile(r'^\s*(\d+)\s*[|\.\):\-]\s*(favor|against|none)\b', re.I)
    proba = np.zeros((N, 3), float)
    got = 0
    for k in range((N + BS - 1) // BS):
        p = f"{F}/opus5_wd_out/b{k}.txt"
        if not os.path.exists(p):
            print(f"[MISS] {p}"); continue
        s = k * BS
        seen = {}
        for ln in open(p, encoding="utf-8"):
            m = LBL.match(ln)
            if m:
                seen[int(m.group(1))] = m.group(2).capitalize()
        e = min(s + BS, N)
        for j in range(e - s):
            lab = seen.get(j + 1, "None")
            proba[s + j, S.LABEL2ID.get(lab, 2)] = 1.0
            got += (j + 1) in seen
    np.savez_compressed(f"{F}/frontier_opus5_test.npz", proba=proba, pred_id=proba.argmax(1))
    print(f"opus5 parsed {got}/{N}  dist={dist(proba.argmax(1))}")
    return proba


def sweep_to_n14(tie):
    base = tie.argmax(1)
    nidx = np.where(base == 2)[0]
    margin = tie[nidx, 2] - np.max(tie[nidx][:, :2], axis=1)  # smaller => weaker None
    keep = set(nidx[np.argsort(-margin)[:14]].tolist())       # keep 14 strongest None
    out = base.copy()
    for i in nidx:
        if i not in keep:
            out[i] = int(np.argmax(tie[i, :2]))                # flip to best non-None
    return out


def build(claude_key, label):
    V = {k: np.load(f"{F}/frontier_{k}_test.npz")["proba"].astype(float)
         for k in ["luna", "g55", "lunaFS", "g55FS"]}
    V["cl"] = np.load(f"{F}/frontier_{claude_key}_test.npz")["proba"].astype(float)
    V["son5"] = np.load(f"{F}/frontier_son5_test.npz")["proba"].astype(float)
    tie = np.stack(list(V.values()), 0).mean(0) + 1e-3 * ce
    out = sweep_to_n14(tie)
    txt = "\n".join(ID2[i] for i in out) + "\n"
    open(f"{F}/pred_{label}.txt", "w").write(txt)
    with zipfile.ZipFile(f"{F}/{label}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("predictions.txt", txt)
    return out


if __name__ == "__main__":
    o5 = assemble_opus5()
    # decorrelation of opus5 vs incumbents
    print("\nopus5 argmax-agreement vs incumbents:")
    for k in ["luna", "g55", "lunaFS", "g55FS", "opus", "son5"]:
        v = np.load(f"{F}/frontier_{k}_test.npz")["proba"].argmax(1)
        print(f"  opus5~{k:7s} {float((o5.argmax(1)==v).mean()):.3f}")
    champ = np.array([S.LABEL2ID[x] for x in
                      zipfile.ZipFile(f"{F}/wd_cl6v_n14.zip").read("predictions.txt").decode().split()])
    my48 = build("opus", "wd_cl6v_my48_n14")   # my-sweep with opus-4-8 (isolates sweep vs champion)
    cand = build("opus5", "wd_cl6v_opus5_n14")  # THE candidate
    print(f"\nchampion(locked)  dist={dist(champ)}")
    print(f"my-sweep opus48   dist={dist(my48)}  diff vs champion={int((my48!=champ).sum())} rows")
    print(f"opus5 candidate   dist={dist(cand)}  diff vs champion={int((cand!=champ).sum())} rows")
    d = np.where(cand != champ)[0]
    print(f"\nopus5 candidate changes {len(d)} rows vs champion (prior A160/F158/N34):")
    for i in d:
        print(f"  row {i}: champ={ID2[champ[i]]:8s} -> opus5cand={ID2[cand[i]]}")
