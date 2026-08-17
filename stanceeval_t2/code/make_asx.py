"""
make_asx.py — build the ArabicStanceX real-multi-target augmentation set (idea-6).

ArabicStanceX (14477 tweets/17 topics, favor/against/none). Real external data = target
diversity for UNSEEN generalization (the encoder-family root-cause fix; distinct from the
failed SYNTHETIC aug — real tweets/labels, no distribution shift). Gate-1: external data
allowed by the competition.

Validation honesty (Gate-2/3): EXCLUDE topics semantically overlapping my 3 eval targets
(Covid Vaccine / Digital Transformation / Women empowerment) so LOTO+dev stay honest:
  Vaccine, vaccine_booster_dose  -> Covid Vaccine
  online_learning                -> Digital Transformation
  women_driving, multi_mariage   -> Women empowerment
Each kept topic gets an Arabic target string (build_prompt shows Arabic → consistent with
the Arabic-glossed competition targets and the Arabic tweets).

Outputs (under datasets/ArabicStanceX/): asx_clean.csv (kept 12 topics) and combined
train_asx.csv = train_track_2 + asx_clean, for --train_csv in train_lora (LOTO stays clean:
no ArabicStanceX target equals a competition target).
"""
import os, sys, json, glob, argparse, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

SRC = "<datasets>/ArabicStanceX_tmp/Stance_task"
OUT = "<datasets>/ArabicStanceX"

EXCLUDE = {"Vaccine", "vaccine_booster_dose", "online_learning", "women_driving", "multi_mariage"}
TARGET_AR = {
    "Aramco": "شركة أرامكو",
    "Neom": "مشروع نيوم",
    "Qiddiya": "مشروع القدية",
    "chineese": "تدريس اللغة الصينية في المدارس",
    "update_curricula": "تحديث المناهج الدراسية",
    "coexistence_with_religions": "التعايش بين الأديان",
    "fix_domestic_tourism": "تنشيط السياحة الداخلية",
    "military_conscription": "التجنيد العسكري الإلزامي",
    "sexual_education_fully": "التثقيف الجنسي في المدارس",
    "mosques_speakers": "مكبرات الصوت في المساجد",
    "Abdulaziz_bin_Turki_Head_of_Sports": "تعيين عبدالعزيز بن تركي رئيساً للرياضة",
    "Faisal_bin_Turkis_resignation": "استقالة فيصل بن تركي",
}
REAL = {"Covid Vaccine", "Digital Transformation", "Women empowerment"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap_per_topic", type=int, default=0, help="0=all; else subsample N/topic")
    ap.add_argument("--out_combined", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synth", "train_asx.csv"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    frames = {}
    for f in glob.glob(f"{SRC}/*/*/*.json"):
        top = os.path.basename(f).replace(".json", "")
        df = pd.DataFrame(json.load(open(f)))
        frames.setdefault(top, []).append(df)

    kept = []
    for top, dfs in frames.items():
        if top in EXCLUDE:
            continue
        assert top in TARGET_AR, f"topic {top} has no Arabic target mapping"
        df = pd.concat(dfs, ignore_index=True)[["text", "stance"]].copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["stance"] = df["stance"].astype(str).str.strip()
        df = df[df["stance"].isin(["Favor", "Against", "None"])]
        df = df[df["text"].str.len() >= 5].drop_duplicates("text")
        df["target"] = TARGET_AR[top]
        if args.cap_per_topic:
            df = df.groupby("stance", group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), max(1, args.cap_per_topic // 3)), random_state=42))
        kept.append(df)
    asx = pd.concat(kept, ignore_index=True)[["text", "target", "stance"]]
    # leakage guard: no ArabicStanceX target may equal a competition eval target
    leak = set(asx["target"]) & REAL
    assert not leak, f"LEAK: {leak}"
    asx.to_csv(f"{OUT}/asx_clean.csv", index=False, encoding="utf-8-sig")

    real = pd.read_csv(S.TRAIN_CSV, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    for c in ("sentiment", "sarcasm"):
        if c not in asx.columns:
            asx[c] = ""
    cols = ["text", "target", "stance", "sentiment", "sarcasm"]
    for c in cols:
        if c not in real.columns:
            real[c] = ""
    combined = pd.concat([real[cols], asx[cols]], ignore_index=True)
    os.makedirs(os.path.dirname(args.out_combined), exist_ok=True)
    combined.to_csv(args.out_combined, index=False, encoding="utf-8-sig")

    print(f"[asx_clean] {len(asx)} rows, {asx['target'].nunique()} topics -> {OUT}/asx_clean.csv")
    print(f"  stance: {dict(asx['stance'].value_counts())}")
    print(f"[combined] real {len(real)} + asx {len(asx)} = {len(combined)} -> {args.out_combined}")
    # validate loadability + LOTO cleanliness
    df = S.load_labeled(args.out_combined, preprocess="light")
    print(f"  load_labeled OK: {len(df)} rows, {df[S.TARGET_COL].nunique()} targets")
    for held in ("Covid Vaccine", "Digital Transformation"):
        ev = (df[S.TARGET_COL] == held).sum(); trn = (df[S.TARGET_COL] != held).sum()
        print(f"  LOTO hold '{held}': eval={ev} (real only) train={trn}")


if __name__ == "__main__":
    main()
