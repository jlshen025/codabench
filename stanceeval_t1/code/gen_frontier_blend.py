"""
gen_frontier_blend.py — blend the frontier-LLM zero-shot voter into the WD ensemble.
Run AFTER llm_predict.py produces _eval/frontier_ds_test.npz. Reuses the cached
Qwen (dep_qens.npz) + CE-enc (enc_ce.npz) proba. Applies the winning Favor→149
calibration. Prints decorrelation + dists; writes candidate zips to submit.
"""
import numpy as np, zipfile, sys
import stance_lib as S
R = "results/eval_deploy_all/_llm"; F = "_eval"
q = np.load(f"{R}/dep_qens.npz")["proba"].astype(float)
enc = np.load(f"{R}/enc_ce.npz")["proba"].astype(float)
fr = np.load(f"{F}/frontier_ds_test.npz")["proba"].astype(float)
ce = 0.15 * enc + 0.85 * q


def dist(p):
    import collections; c = collections.Counter(p.argmax(1).tolist()); return {S.ID2LABEL[k]: c.get(k, 0) for k in range(3)}


def calib_favor(p, target=149):
    lo, hi = -0.8, 0.4
    for _ in range(60):
        mid = (lo + hi) / 2
        if int(((p + np.array([0, mid, 0])).argmax(1) == 1).sum()) > target:
            hi = mid
        else:
            lo = mid
    return p + np.array([0, (lo + hi) / 2, 0])


def savezip(p, name):
    lab = [S.ID2LABEL[i] for i in p.argmax(1)]
    open(f"{F}/pred_{name}.txt", "w").write("\n".join(lab) + "\n")
    with zipfile.ZipFile(f"{F}/{name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(f"{F}/pred_{name}.txt", "predictions.txt")
    print(f"  {name}: {dist(p)}")


print("frontier dist", dist(fr), "| ce dist", dist(ce), "| qwen dist", dist(q))
print("argmax-agree  ce~frontier", round(float(np.mean(ce.argmax(1) == fr.argmax(1))), 3),
      " qwen~frontier", round(float(np.mean(q.argmax(1) == fr.argmax(1))), 3))
# frontier SOLO (calibrated) + ce+frontier blends at a few weights (calibrated Favor->149)
savezip(calib_favor(fr, 149), "fr_solo_cal")
for wf in [0.15, 0.25, 0.35]:
    savezip(calib_favor((1 - wf) * ce + wf * fr, 149), f"cf{int(wf*100)}_cal")
print("DONE — candidates: fr_solo_cal, cf15_cal, cf25_cal, cf35_cal (all Favor→149)")
