"""
stance_lib.py — Core library for StanceEval-2026 (Mawqif-v2 Arabic stance detection).

Single-sourced definitions used by every training / eval / submission script:
  * label maps (MUST match the official Evaluation_script.ipynb)
  * robust CSV loading (keep_default_na=False so the stance "None" stays a STRING)
  * Arabic preprocessing (baseline replica + a "light" variant that keeps emojis/Latin)
  * the EXACT official metric (Favg2 = mean(F1_Favor, F1_Against), pooled)
  * CV splitters: Leave-One-Target-Out (Track-2 proxy) and stratified k-fold (Track-1)

Provenance of the metric: doc/Evaluation_script.ipynb compute_metrics() — verbatim.
"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATA_DIR = os.environ.get("MAWQIF_DIR", "<datasets>/MawqifV2")
# Track-2 (UNSEEN) split (fetched 2026-07-01 from repo MawqifV2/Track 2/):
#   train_track_2.csv = 2721 rows, SEEN targets {Covid Vaccine, Digital Transformation}
#   dev_track_2.csv   = 1400 rows, UNSEEN target {Women empowerment} (fully labeled) — the Dev board scores THIS
TRAIN_CSV = os.path.join(DATA_DIR, "train_track_2.csv")
DEV_CSV = os.path.join(DATA_DIR, "dev_track_2.csv")

# Order MUST match the official scorer: Against=0, Favor=1, None=2
LABEL2ID = {"Against": 0, "Favor": 1, "None": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
VALID_LABELS = ["Against", "Favor", "None"]

TEXT_COL = "text"
TARGET_COL = "target"
LABEL_COL = "stance"

# The SEEN targets present in Track-2 train (exact strings as in the CSV).
# Note: "Women empowerment" is now the UNSEEN dev target (moved out of training).
SEEN_TARGETS = ["Covid Vaccine", "Digital Transformation"]

# Short Arabic background descriptions per target (ZSSD "knowledge injection": a target
# gloss bridges SEEN->UNSEEN targets — SIGIR'22 + 2024-25 SOTA). Used via
# train.py/predict --target_template "{target}: {desc}" (key kn_ar). ADD the blind-test
# targets here at eval time (fetch from Arabic Wikipedia / write a 1-line gloss).
TARGET_DESCS = {
    "Covid Vaccine": "لقاح كوفيد-19 للوقاية من فيروس كورونا المستجد",
    "Digital Transformation": "التحول الرقمي وإدخال التقنيات الرقمية في الخدمات والأعمال والحكومة",
    "Women empowerment": "تمكين المرأة ومنحها حقوق وفرص متساوية في التعليم والعمل والمجتمع",
}

# Auxiliary label maps (for multi-task learning).
SENT2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
SARC2ID = {"No": 0, "Yes": 1}

# --------------------------------------------------------------------------- #
# Arabic preprocessing
# --------------------------------------------------------------------------- #
_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰـ]")  # tashkeel + tatweel
_BASELINE_DIAC = re.compile(r"ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ")
_NON_ARABIC = re.compile(r"[^؀-ۿ0-9\s]+")
_MULTI_SPACE = re.compile(r"\s+")
_REPEATED = re.compile(r"(.)\1{2,}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")


def preprocess_baseline(text: str) -> str:
    """Exact replica of the organizers' baseline preprocessing (strips all non-Arabic)."""
    text = str(text)
    text = _BASELINE_DIAC.sub("", text)
    text = _NON_ARABIC.sub(" ", text)
    text = _REPEATED.sub(r"\1\1", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def _normalize_arabic(text: str) -> str:
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return text


def preprocess_light(text: str, normalize=False) -> str:
    """Lighter cleaning: drop URLs/mentions/diacritics, squeeze repeats, KEEP emojis/Latin/hashwords."""
    text = str(text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", " ").replace("_", " ")
    text = _DIACRITICS.sub("", text)
    if normalize:
        text = _normalize_arabic(text)
    text = _REPEATED.sub(r"\1\1", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


PREPROCESSORS = {
    "none": lambda t: str(t).strip(),
    "baseline": preprocess_baseline,
    "light": preprocess_light,
    "light_norm": lambda t: preprocess_light(t, normalize=True),
}

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_labeled(path: str, preprocess: str = "baseline") -> pd.DataFrame:
    """Load a labeled train/dev CSV. keep_default_na=False keeps stance 'None' as a string."""
    df = pd.read_csv(path, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    for c in (TEXT_COL, TARGET_COL, LABEL_COL):
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
        df[c] = df[c].astype(str).str.strip()
    df = df[(df[TEXT_COL] != "") & (df[TARGET_COL] != "") & (df[LABEL_COL] != "")].copy()
    unknown = sorted(set(df[LABEL_COL]) - set(LABEL2ID))
    if unknown:
        raise ValueError(f"Unknown stance labels: {unknown}")
    fn = PREPROCESSORS[preprocess]
    df["text_proc"] = df[TEXT_COL].apply(fn)
    df["label"] = df[LABEL_COL].map(LABEL2ID).astype(int)
    # aux labels (may be blank for some rows) -> -100 = ignore
    if "sentiment" in df.columns:
        df["sent_label"] = df["sentiment"].map(SENT2ID).fillna(-100).astype(int)
    if "sarcasm" in df.columns:
        df["sarc_label"] = df["sarcasm"].map(SARC2ID).fillna(-100).astype(int)
    return df.reset_index(drop=True)


def load_unlabeled(path: str, preprocess: str = "baseline") -> pd.DataFrame:
    """Load an unlabeled test CSV (id, text, target)."""
    df = pd.read_csv(path, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    id_col = "id" if "id" in df.columns else ("ID" if "ID" in df.columns else None)
    for c in (TEXT_COL, TARGET_COL):
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
        df[c] = df[c].astype(str).str.strip()
    fn = PREPROCESSORS[preprocess]
    df["text_proc"] = df[TEXT_COL].apply(fn)
    if id_col:
        df["id"] = df[id_col].astype(str).str.strip()
    return df.reset_index(drop=True)

# --------------------------------------------------------------------------- #
# Metric — EXACT replica of doc/Evaluation_script.ipynb
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred) -> dict:
    """y_true / y_pred are integer label ids (0=Against,1=Favor,2=None). Returns fractions in [0,1]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f_against = f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    f_favor = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    f_none = f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    favg2 = (f_favor + f_against) / 2.0
    favg3 = (f_favor + f_against + f_none) / 3.0
    return {
        "F_favor": float(f_favor), "F_against": float(f_against), "F_none": float(f_none),
        "Favg2": float(favg2), "Favg3": float(favg3), "Acc": float(accuracy_score(y_true, y_pred)),
    }


def metrics_from_labels(gold_labels, pred_labels) -> dict:
    """Convenience: accept string labels ('Favor'/'Against'/'None')."""
    yt = [LABEL2ID[x] for x in gold_labels]
    yp = [LABEL2ID[x] for x in pred_labels]
    return compute_metrics(yt, yp)


def favg2_from_proba(proba, y_true, thresholds=None) -> dict:
    """Apply per-class additive thresholds to class probabilities, argmax, then score.
    thresholds: array len-3 added to log/probabilities before argmax (default zeros).
    """
    proba = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.zeros(proba.shape[1])
    pred = np.argmax(proba + np.asarray(thresholds), axis=1)
    m = compute_metrics(y_true, pred)
    m["pred"] = pred
    return m

# --------------------------------------------------------------------------- #
# CV splitters
# --------------------------------------------------------------------------- #
def leave_one_target_out(df: pd.DataFrame):
    """Yield (train_idx, val_idx, held_target). Mirrors the Track-2 unseen-target protocol."""
    targets = list(dict.fromkeys(df[TARGET_COL].tolist()))  # preserve order, unique
    for t in targets:
        val_idx = df.index[df[TARGET_COL] == t].to_numpy()
        train_idx = df.index[df[TARGET_COL] != t].to_numpy()
        yield train_idx, val_idx, t


def stratified_kfold_indices(df: pd.DataFrame, n_splits=5, seed=42, by="stance_target"):
    """Stratified k-fold indices. by='stance' or 'stance_target' (joint strata)."""
    from sklearn.model_selection import StratifiedKFold
    if by == "stance_target":
        strat = (df[LABEL_COL].astype(str) + "|" + df[TARGET_COL].astype(str)).to_numpy()
    else:
        strat = df[LABEL_COL].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(df))
    for tr, va in skf.split(dummy, strat):
        yield df.index.to_numpy()[tr], df.index.to_numpy()[va]


if __name__ == "__main__":
    # Self-test: load data, print stats, verify metric on a trivial case, show CV fold sizes.
    tr = load_labeled(TRAIN_CSV)
    dv = load_labeled(DEV_CSV)
    print(f"train={tr.shape} dev={dv.shape}")
    print("train targets:", dict(tr[TARGET_COL].value_counts()))
    print("train stance :", dict(tr[LABEL_COL].value_counts()))
    # metric sanity: perfect prediction -> Favg2=1.0
    perfect = compute_metrics(tr["label"], tr["label"])
    print("perfect Favg2:", round(perfect["Favg2"], 4), "(expect 1.0)")
    # majority-class (all Favor=1) baseline on dev
    maj = compute_metrics(dv["label"], np.ones(len(dv), dtype=int))
    print("all-Favor dev Favg2:", round(maj["Favg2"] * 100, 2), "Favg3:", round(maj["Favg3"] * 100, 2))
    print("LOTO folds:")
    for trn, val, t in leave_one_target_out(tr):
        print(f"  hold '{t}': train={len(trn)} val={len(val)}")
