"""
feat_lr.py — a DECORRELATED non-neural stance component: TF-IDF (word+char) + target
one-hot → Logistic Regression. Different inductive bias than the neural encoders/LLMs
(bag-of-ngrams + linear), so its errors should decorrelate → real ensemble diversity
(the neural voters are all 0.86-0.96 pred-correlated; this is the missing orthogonal signal).

Produces dev-OOF proba (train→dev, aligned to dev row order) in stance_lib column order
[Against,Favor,None] → directly blendable with results/*/oof_proba.npz. CPU-only.
C tuned on TRAIN k-fold only (no dev peeking).
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack


def build_features(tr_text, dv_text, tr_tgt, dv_tgt, word_ng, char_ng):
    wv = TfidfVectorizer(analyzer="word", ngram_range=word_ng, min_df=2, sublinear_tf=True)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ng, min_df=3, sublinear_tf=True)
    Xw_tr, Xw_dv = wv.fit_transform(tr_text), wv.transform(dv_text)
    Xc_tr, Xc_dv = cv.fit_transform(tr_text), cv.transform(dv_text)
    oh = OneHotEncoder(handle_unknown="ignore")
    Tt = oh.fit_transform(np.array(tr_tgt).reshape(-1, 1))
    Td = oh.transform(np.array(dv_tgt).reshape(-1, 1))
    return hstack([Xw_tr, Xc_tr, Tt]).tocsr(), hstack([Xw_dv, Xc_dv, Td]).tocsr(), (Xw_tr.shape[1], Xc_tr.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess", default="light_norm")
    ap.add_argument("--out_npz", default="scripts/_llm/featlr_dev.npz")
    ap.add_argument("--word_ng", default="1,2")
    ap.add_argument("--char_ng", default="2,5")
    ap.add_argument("--class_weight", default="none", choices=["none", "balanced"])
    args = ap.parse_args()
    word_ng = tuple(int(x) for x in args.word_ng.split(","))
    char_ng = tuple(int(x) for x in args.char_ng.split(","))
    cw = None if args.class_weight == "none" else "balanced"

    tr = S.load_labeled(S.TRAIN_CSV, preprocess=args.preprocess)
    dv = S.load_labeled(S.DEV_CSV, preprocess=args.preprocess)
    Xtr, Xdv, dims = build_features(tr["text_proc"], dv["text_proc"],
                                    tr[S.TARGET_COL], dv[S.TARGET_COL], word_ng, char_ng)
    ytr, ydv = tr["label"].to_numpy(), dv["label"].to_numpy()
    tgt = dv[S.TARGET_COL].to_numpy().astype(str)
    print(f"[feat] word={dims[0]} char={dims[1]} +3 target = {Xtr.shape[1]} dims", flush=True)

    # tune C on TRAIN 5-fold (Favg2), no dev peeking
    strat = (tr[S.LABEL_COL].astype(str) + "|" + tr[S.TARGET_COL].astype(str)).to_numpy()
    best_C, best_cv = None, -1
    for C in [0.3, 1.0, 3.0, 10.0, 30.0]:
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        oof = np.zeros(len(tr), int)
        for a, b in skf.split(np.zeros(len(tr)), strat):
            clf = LogisticRegression(C=C, max_iter=3000, class_weight=cw)
            clf.fit(Xtr[a], ytr[a])
            oof[b] = clf.predict(Xtr[b])
        f = S.compute_metrics(ytr, oof)["Favg2"] * 100
        if f > best_cv:
            best_cv, best_C = f, C
    print(f"[C-tune on train] best C={best_C} (train-CV Favg2={best_cv:.2f})", flush=True)

    clf = LogisticRegression(C=best_C, max_iter=5000, class_weight=cw)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xdv)          # columns = clf.classes_ = [0,1,2] = [Against,Favor,None]
    assert list(clf.classes_) == [0, 1, 2]
    m = S.compute_metrics(ydv, proba.argmax(1))
    per = {t.split()[0]: round(S.compute_metrics(ydv[tgt == t], proba[tgt == t].argmax(1))["Favg2"] * 100, 1)
           for t in sorted(set(tgt.tolist()))}
    print(f"[DEV solo] Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f} Acc={m['Acc']*100:.2f} per-target={per}", flush=True)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out_npz) \
        if not os.path.isabs(args.out_npz) else args.out_npz
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, proba=proba.astype(np.float32), y_true=ydv, target=tgt)
    print(f"[out] -> {out}", flush=True)


if __name__ == "__main__":
    main()
