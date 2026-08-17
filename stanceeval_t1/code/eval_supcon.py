"""
eval_supcon.py — judge the SupCon encoder lever ROBUSTLY (not by the seed-lucky [42,1] draw).

Compares CE-MTL encoders vs SupCon-MTL encoders (per λ) on:
  * encoder solo Favg2 + per-target (vs CE 85.09)
  * per-(model,seed) solo spread  = seed-variance (Gunel's robustness claim)
  * ⊕ Qwen-ens[42,1]  nested-CV   (vs held-best 89.78; contaminated by [42,1] seed-luck)
  * ⊕ Qwen-ens[4seed] nested-CV   (vs seed-expected 88.10; the ROBUST test)
  * MIX (CE ⊕ SupCon encoders)    (diversity play)
Promote ONLY a lever that lifts the ROBUST blend (⊕Qwen4 > 88.10 gauge-robust) and/or the solo.
"""
import numpy as np, blend_ens as BE, stance_lib as S
L = BE.L


def load_enc(dirs):
    P, y, t = [], None, None
    for d in dirs:
        z = np.load(f"{L}/{d}/oof_proba.npz", allow_pickle=True)
        P.append(z["proba"].astype(float))
        if y is None:
            y, t = z["y_true"].astype(int), z["target"].astype(str)
    return np.mean(P, 0), y, t


def f2(y, p, m=None):
    yy, pp = (y, p) if m is None else (y[m], p[m])
    return S.compute_metrics(yy, pp.argmax(1))["Favg2"] * 100


def pt(y, p, t):
    return {x.split()[0]: round(f2(y, p, t == x), 1) for x in sorted(set(t.tolist()))}


enc_ce, y, t = BE.enc_comp()
qwen2 = BE.llm_ens(["qwen14", "q14lat"], [42, 1], 3)
qwen4 = BE.llm_ens(["qwen14", "q14lat"], [42, 1, 7, 13], 3)

ce_dirs = [f"mtl_{m}_dev_s{s}" for m in ["marbertv2", "araberttw"] for s in [1, 42]]
ce_solos = [f2(y, load_enc([d])[0]) for d in ce_dirs]
b2 = BE.nested_cv(enc_ce, qwen2, y, t); b4 = BE.nested_cv(enc_ce, qwen4, y, t)
print(f"CE-enc(4x) solo {f2(y, enc_ce):.2f}  {pt(y, enc_ce, t)}")
print(f"  per-(model,seed) solos {[round(x,2) for x in ce_solos]}  spread={max(ce_solos)-min(ce_solos):.2f}")
print(f"  BASELINE ⊕Qwen[42,1] {b2[0]:.2f}±{b2[1]:.2f} (=89.78 held-best)   ⊕Qwen4 {b4[0]:.2f}±{b4[1]:.2f} (=88.10 seed-expected)")

for lam in ["03", "10"]:
    dirs = [f"mtl_{m}_sc{lam}_s{s}" for m in ["marbertv2", "araberttw"] for s in [1, 42]]
    enc_sc, _, _ = load_enc(dirs)
    sc_solos = [f2(y, load_enc([d])[0]) for d in dirs]
    m2 = BE.nested_cv(enc_sc, qwen2, y, t); m4 = BE.nested_cv(enc_sc, qwen4, y, t)
    enc_mix = 0.5 * enc_ce + 0.5 * enc_sc
    x2 = BE.nested_cv(enc_mix, qwen2, y, t); x4 = BE.nested_cv(enc_mix, qwen4, y, t)
    print(f"\n=== SupCon λ={ {'03':'0.3','10':'1.0'}[lam] } ===")
    print(f"  supcon-enc(4x) solo {f2(y, enc_sc):.2f}  {pt(y, enc_sc, t)}   (CE 85.09, Δ={f2(y,enc_sc)-85.09:+.2f})")
    print(f"  per-(model,seed) solos {[round(x,2) for x in sc_solos]}  spread={max(sc_solos)-min(sc_solos):.2f}  (CE spread {max(ce_solos)-min(ce_solos):.2f})")
    print(f"  ⊕Qwen[42,1] {m2[0]:.2f}±{m2[1]:.2f} (vs 89.78, Δ={m2[0]-89.78:+.2f})   ⊕Qwen4 {m4[0]:.2f}±{m4[1]:.2f} (vs 88.10 ROBUST, Δ={m4[0]-b4[0]:+.2f})")
    print(f"  MIX(CE⊕SupCon) ⊕Qwen[42,1] {x2[0]:.2f} (Δ={x2[0]-89.78:+.2f})   ⊕Qwen4 {x4[0]:.2f} (Δ={x4[0]-b4[0]:+.2f})")
