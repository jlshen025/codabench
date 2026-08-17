"""
ens_frontier.py — build frontier-ensemble WD candidates from voter npzs.

Voters are zero-shot LLM proba npzs (one-hot or soft) aligned to test_seen_norm.csv
row order. Prints the pairwise decorrelation matrix + dists, then writes candidate
zips: each solo, an equal soft-average, and a majority vote with a weak tiebreak
(cached ce proba) for the disagreement rows. Favg2 excludes None, so None calls
still matter (they remove Favor/Against false positives).

Usage: python ens_frontier.py sol=_eval/frontier_ds_test.npz goss=_eval/frontier_gptoss_test.npz [luna=...]
"""
import sys, os, zipfile, itertools, collections
import numpy as np
import stance_lib as S

F = "_eval"; R = "results/eval_deploy_all/_llm"
ce = (0.15 * np.load(f"{R}/enc_ce.npz")["proba"] + 0.85 * np.load(f"{R}/dep_qens.npz")["proba"]).astype(float)


def dist(a):
    c = collections.Counter(a.tolist()); return {S.ID2LABEL[k]: c.get(k, 0) for k in range(3)}


def savez(a, name):
    open(f"{F}/pred_{name}.txt", "w").write("\n".join(S.ID2LABEL[i] for i in a) + "\n")
    with zipfile.ZipFile(f"{F}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{F}/pred_{name}.txt", "predictions.txt")
    return dist(a)


def main():
    voters = {}
    for arg in sys.argv[1:]:
        k, p = arg.split("=", 1)
        voters[k] = np.load(p)["proba"].astype(float)
    names = list(voters)
    print("TRUE prior ~ Against160 / Favor158 / None34")
    for k in names:
        print(f"  {k:8s} dist {dist(voters[k].argmax(1))}")
    print("pairwise argmax-agree:")
    for a, b in itertools.combinations(names, 2):
        ag = float((voters[a].argmax(1) == voters[b].argmax(1)).mean())
        print(f"  {a}~{b}: {ag:.3f}")
    # solos
    for k in names:
        print(f"solo_{k:8s}:", savez(voters[k].argmax(1), f"ens_solo_{k}"))
    # equal soft-average of all voters (one-hot averaged); ties broken by ce
    stack = np.stack([voters[k] for k in names], 0).mean(0)  # [N,3]
    tie_eps = 1e-3 * ce  # tiny ce nudge to break exact ties toward the calibrated blend
    avg = (stack + tie_eps).argmax(1)
    print("soft_avg      :", savez(avg, "ens_softavg"))
    # majority with ce tiebreak on disagreement (only meaningful for >=2 voters)
    if len(names) >= 2:
        preds = np.stack([voters[k].argmax(1) for k in names], 0)  # [V,N]
        out = np.zeros(preds.shape[1], dtype=int)
        for j in range(preds.shape[1]):
            cnt = collections.Counter(preds[:, j].tolist())
            top = cnt.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                out[j] = top[0][0]
            else:  # tie -> ce argmax among tied classes
                tied = [c for c, n in top if n == top[0][1]]
                out[j] = tied[int(np.argmax([ce[j, c] for c in tied]))]
        print("majority_ce   :", savez(out, "ens_majce"))
    # sol-anchored: keep sol, but where ALL others agree AND differ from sol, flip to them
    if "sol" in voters and len(names) >= 2:
        sol = voters["sol"].argmax(1); others = [voters[k].argmax(1) for k in names if k != "sol"]
        oth = np.stack(others, 0)
        allagree = (oth == oth[0]).all(0)
        flip = allagree & (oth[0] != sol)
        anch = sol.copy(); anch[flip] = oth[0][flip]
        print(f"sol_anchored  (flips {int(flip.sum())}):", savez(anch, "ens_solanch"))


if __name__ == "__main__":
    main()
