"""
build_ens.py — analyze LLM-voter decorrelation on the BLIND test + build majority
ensembles and write submission .txt files (file order, exact labels).

Test has no gold → we cannot score locally; this only measures voter AGREEMENT
(decorrelation potential vs the held best sol=90.16) and materializes candidate
.txt files to submit. ID map: Against=0, Favor=1, None=2.
"""
import numpy as np, os, sys, itertools

D = "_llm"
ID2L = {0: "Against", 1: "Favor", 2: "None"}
VOTERS = {
    "sol":   "g56fix_grounded_test",   # held best = 90.16 (857582)
    "dspro": "dspro_grounded_test",
    "luna":  "luna_grounded_test",
    "terra": "terra_grounded_test",
    "dsflash": "dsflash_grounded_test",  # 86.49 anchor (weaker)
}

P, PRED = {}, {}
IDX0 = TGT = None
for k, f in VOTERS.items():
    z = np.load(f"{D}/{f}.npz", allow_pickle=True)
    P[k] = z["proba"].astype(float)
    PRED[k] = P[k].argmax(1)
    idx = z["idx"]; tgt = z["target"].astype(str)
    if IDX0 is None:
        IDX0, TGT = idx, tgt
    else:
        assert np.array_equal(idx, IDX0), f"{k} idx order differs!"
N = len(IDX0)
targets = list(dict.fromkeys(TGT.tolist()))
print(f"N={N} targets={targets}")

print("\n=== per-voter label dist (Against/Favor/None) overall + per target ===")
for k in VOTERS:
    row = [f"{k:8s}"]
    for scope, mask in [("all", np.ones(N, bool))] + [(t, TGT == t) for t in targets]:
        c = np.bincount(PRED[k][mask], minlength=3)
        row.append(f"{scope}:A{c[0]}/F{c[1]}/N{c[2]}")
    print("  " + "  ".join(row))

print("\n=== argmax agreement with sol (overall + per target) ===")
for k in VOTERS:
    if k == "sol":
        continue
    row = [f"{k:8s} all=%.3f" % (PRED[k] == PRED["sol"]).mean()]
    for t in targets:
        m = TGT == t
        row.append(f"{t}=%.3f" % (PRED[k][m] == PRED["sol"][m]).mean())
    print("  " + "  ".join(row))


def majority(keys, tiebreak="sol"):
    """Sum one-hot votes; ties -> tiebreak voter's pred."""
    votes = np.zeros((N, 3))
    for k in keys:
        votes += P[k]
    out = votes.argmax(1)
    mx = votes.max(1, keepdims=True)
    tie = (votes == mx).sum(1) > 1   # >1 class tied at the max
    out[tie] = PRED[tiebreak][tie]
    return out


CANDS = {
    "sol":       ["sol"],
    "dspro":     ["dspro"],
    "luna":      ["luna"],
    "terra":     ["terra"],
    "maj_sdl":   ["sol", "dspro", "luna"],
    "maj_sdt":   ["sol", "dspro", "terra"],
    "maj_slt":   ["sol", "luna", "terra"],       # gpt5.6 sibling consensus
    "maj_sdlt_d": ["sol", "dspro", "luna", "terra"],  # 4-way, tie->dspro? no, tie->sol
    "maj5":      ["sol", "dspro", "luna", "terra", "dsflash"],
}
print("\n=== candidate ensembles: #rows differing from sol (per target) ===")
os.makedirs("staging", exist_ok=True)
for name, keys in CANDS.items():
    pred = PRED[keys[0]] if len(keys) == 1 else majority(keys)
    diff = pred != PRED["sol"]
    per = "  ".join(f"{t}:{int((diff & (TGT==t)).sum())}" for t in targets)
    print(f"  {name:12s} keys={'+'.join(keys):28s} diff_sol={int(diff.sum()):3d}  ({per})")
    # write txt (file order)
    with open(f"staging/e_{name}.txt", "w") as fh:
        fh.write("\n".join(ID2L[int(p)] for p in pred) + "\n")
print("\n[wrote staging/e_*.txt]")
