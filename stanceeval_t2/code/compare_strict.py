"""Baseline vs strict-None sol-grounded on the GOLD proxy, per target. Order-independent
(each npz scored within its own target mask). GATE: strict robustly >= baseline on ALL targets."""
import numpy as np
import stance_lib as S

def per_target(f):
    z = np.load(f, allow_pickle=True)
    p, t, y = z["pred_id"], z["target"].astype(str), z["y_true"]
    out = {}
    for tg in ["ALL"] + list(dict.fromkeys(t.tolist())):
        m = np.ones(len(y), bool) if tg == "ALL" else (t == tg)
        out[tg] = (round(S.compute_metrics(y[m], p[m])["Favg2"] * 100, 2),
                   round((p[m] == 2).mean() * 100, 1), round((y[m] == 2).mean() * 100, 1))
    return out

PAIRS = [("WomenEmp-dev", "_llm/g56_grounded_dev.npz", "_llm/sol_strict_dev.npz"),
         ("train-LOTO",   "_llm/sol_train_loto.npz",   "_llm/sol_strict_train.npz")]
print(f"{'proxy/target':28s} {'baseF2':>7s} {'strictF2':>8s} {'Δ':>7s}  predNone base→strict (true)")
deltas = []
for name, bf, sf in PAIRS:
    try:
        B, Sc = per_target(bf), per_target(sf)
    except FileNotFoundError as e:
        print(f"  [missing] {name}: {e}"); continue
    for tg in B:
        b, s = B[tg], Sc.get(tg)
        if s is None: continue
        d = round(s[0] - b[0], 2)
        if tg != "ALL":
            deltas.append((f"{name}/{tg}", d))
        print(f"  {name+'/'+tg:26s} {b[0]:7.2f} {s[0]:8.2f} {d:+7.2f}  {b[1]:4.1f}%→{s[1]:4.1f}% (true {b[2]:4.1f}%)")
print("\n=== per-target deltas (the GATE) ===")
for k, d in deltas:
    print(f"  {k:28s} {d:+.2f}  {'OK' if d >= -0.3 else 'REGRESSION'}")
worst = min(d for _, d in deltas) if deltas else None
print(f"\nworst per-target Δ = {worst:+.2f}  → strict-None {'ROBUST (go to test)' if worst is not None and worst >= -0.3 else 'TARGET-SPECIFIC / hurts (do NOT deploy)'}")
