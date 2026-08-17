"""
make_combined.py — build a train+synthetic combined CSV for LoRA retraining (idea-4).

Concats train_track_2.csv (2721 real) + synth_aim.csv (constructed aim-targeted synthetic
on DIVERSE FAKE Arabic targets). Because synth targets never equal Covid Vaccine /
Digital Transformation / Women empowerment, `--hold_target` LOTO in train_lora stays clean
(synth rows always land in TRAIN, never in the held-out eval fold).

Guards: refuses if any synth target collides with a real/eval target, or any synth stance
is invalid. Output columns = text,target,stance,sentiment,sarcasm (aux blank for synth;
LoRA ignores aux, MTL-safe anyway).
"""
import os, sys, argparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

REAL_TARGETS = {"Covid Vaccine", "Digital Transformation", "Women empowerment"}
VALID = {"Against", "Favor", "None"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synth", "synth_aim.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synth", "train_synth_aim.csv"))
    ap.add_argument("--max_per_scenario", type=int, default=0, help="0=all; else cap rows per scenario (ablation)")
    ap.add_argument("--scenarios", default="", help="comma-list to KEEP (default all)")
    args = ap.parse_args()

    real = pd.read_csv(S.TRAIN_CSV, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    syn = pd.read_csv(args.synth, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    for c in ("text", "target", "stance"):
        syn[c] = syn[c].astype(str).str.strip()

    # ---- guards ----
    leak = set(syn["target"]) & REAL_TARGETS
    assert not leak, f"SYNTH TARGET LEAK: {leak}"
    bad = set(syn["stance"]) - VALID
    assert not bad, f"invalid synth stance: {bad}"

    if args.scenarios:
        keep = set(s.strip() for s in args.scenarios.split(","))
        syn = syn[syn["scenario"].isin(keep)].copy()
    if args.max_per_scenario:
        syn = syn.groupby("scenario", group_keys=False).head(args.max_per_scenario).copy()

    # ---- align columns ----
    keep_cols = ["text", "target", "stance", "sentiment", "sarcasm"]
    for c in keep_cols:
        if c not in real.columns:
            real[c] = ""
        if c not in syn.columns:
            syn[c] = ""
    out = pd.concat([real[keep_cols], syn[keep_cols]], ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[combined] real={len(real)} + synth={len(syn)} = {len(out)} -> {args.out}")
    print(f"  real stance : {dict(real['stance'].value_counts())}")
    print(f"  synth stance: {dict(syn['stance'].value_counts())}")
    print(f"  synth targets: {syn['target'].nunique()} unique (all fake, LOTO-clean)")
    # verify load_labeled reads it & LOTO folds are clean
    df = S.load_labeled(args.out, preprocess="light")
    print(f"  load_labeled OK: {len(df)} rows, targets={sorted(df[S.TARGET_COL].unique())[:3]}... ({df[S.TARGET_COL].nunique()} total)")
    for held in ("Covid Vaccine", "Digital Transformation"):
        ev = (df[S.TARGET_COL] == held).sum()
        trn = (df[S.TARGET_COL] != held).sum()
        print(f"  LOTO hold '{held}': eval={ev} (real only) train={trn} (incl synth)")


if __name__ == "__main__":
    main()
