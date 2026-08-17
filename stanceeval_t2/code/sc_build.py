"""self-consistency: combine 5 temp0.7 gpt-5.6-sol samples vs the greedy sol=90.16."""
import numpy as np
D = "_llm"; ID2L = {0: "Against", 1: "Favor", 2: "None"}
def load(f):
    z = np.load(f"{D}/{f}.npz", allow_pickle=True)
    return z["proba"].astype(float), z["target"].astype(str)
Psol, TGT = load("g56fix_grounded_test")
SOL = Psol.argmax(1)
scs = {f"sc{i}": load(f"sc{i}_test")[0] for i in range(1, 6)}
N = len(SOL); targets = list(dict.fromkeys(TGT.tolist()))
print(f"N={N}")
print("=== each temp0.7 sample: agree w/ sol-greedy + label dist ===")
for k, p in scs.items():
    pr = p.argmax(1)
    c = np.bincount(pr, minlength=3)
    print(f"  {k}: agree_sol=%.3f  A{c[0]}/F{c[1]}/N{c[2]}" % (pr == SOL).mean())

def maj(mats, tiebreak=None):
    v = np.zeros((N, 3))
    for m in mats:
        v += (m == m.max(1, keepdims=True)) / (m == m.max(1, keepdims=True)).sum(1, keepdims=True) if m.ndim == 2 else 0
    # mats are one-hot proba already -> just sum
    v = sum(mats)
    out = v.argmax(1)
    if tiebreak is not None:
        tie = (v == v.max(1, keepdims=True)).sum(1) > 1
        out[tie] = tiebreak[tie]
    return out

cands = {
    "sc5":   maj(list(scs.values())),                       # pure SC {sc1..5}
    "scG":   maj([Psol] + list(scs.values()), tiebreak=SOL),  # {sol+sc1..5} tie->sol
    "scG3":  maj([Psol] + list(scs.values())[:2], tiebreak=SOL),  # {sol+sc1+sc2}
}
print("\n=== SC candidates: diff-from-sol (per target) ===")
for name, pred in cands.items():
    diff = pred != SOL
    per = "  ".join(f"{t}:{int((diff&(TGT==t)).sum())}" for t in targets)
    print(f"  {name:5s} diff_sol={int(diff.sum()):3d}  ({per})")
    with open(f"staging/e_{name}.txt", "w") as fh:
        fh.write("\n".join(ID2L[int(p)] for p in pred) + "\n")
print("[wrote staging/e_sc*.txt]")
