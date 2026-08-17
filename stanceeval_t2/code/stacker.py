#!/usr/bin/env python
"""stacker.py — supervised meta-learner over independent readers' votes -> gold.

Idea (named by the blinded 08-03 consult, E58): model the ANNOTATOR'S CONVENTION rather
than the truth. Features are the per-row votes of several independent zero-shot readers;
the target is the gold label. A stacker can learn the systematic offset between a model's
natural reading and this corpus's convention (e.g. "when reader A says None and reader B
says Favor, the annotator wrote Favor") — which equal-weight majority voting, already
ruled out here, cannot express.

Validation is LEAVE-ONE-TARGET-OUT over the 3 labelled targets, because the competition
test targets are unseen; a random split would be measuring the wrong thing entirely.
Features are deliberately target-agnostic (votes only, no lexical features) so nothing
target-specific can leak into a model that must generalise to new topics.

Reported against the two baselines that matter: the BEST SINGLE reader on the same fold,
and plain majority vote. A stacker that cannot beat its own best input is worthless.
"""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0, '.')
import stance_lib as S
from sklearn.linear_model import LogisticRegression

NAMES = np.array(['Against', 'Favor', 'None'])
READERS = [('opus5', 'o5_train', 'o5_dev_ctrl'),
           ('sonnet5', 's5_train', 's5_dev'),
           ('claude', 'claude_grounded_loto', 'claude_grounded_devt2'),
           ('dsflash', 'dsflash_grounded_loto', 'dsflash_grounded_devt2'),
           ('dspro', 'dspro_grounded_loto', 'dspro_grounded_devt2')]


def load_split(fname, csv):
    """Return (pred_id, y_true, target) re-ordered to the CSV's own row order via idx."""
    z = np.load(f'_llm/{fname}.npz', allow_pickle=True)
    idx = z['idx']
    p = z['pred_id'] if 'pred_id' in z else z['proba'].argmax(1)
    n = len(pd.read_csv(csv, keep_default_na=False, dtype=str))
    out = np.full(n, -1, dtype=int)
    out[idx] = p
    y = np.full(n, -1, dtype=int)
    if 'y_true' in z:
        y[idx] = z['y_true']
    return out, y


def build():
    tr = pd.read_csv(S.TRAIN_CSV, keep_default_na=False, encoding='utf-8-sig')
    dv = pd.read_csv(S.DEV_CSV, keep_default_na=False, encoding='utf-8-sig')
    tgt = np.concatenate([tr['target'].values, dv['target'].values])
    gold = np.array([S.LABEL2ID[s.strip()] for s in
                     np.concatenate([tr['stance'].values, dv['stance'].values])])
    V = {}
    for nm, ftr, fdv in READERS:
        a, _ = load_split(ftr, S.TRAIN_CSV)
        b, _ = load_split(fdv, S.DEV_CSV)
        V[nm] = np.concatenate([a, b])
    return tgt, gold, V


def feats(V, rows):
    """One-hot each reader's vote -> 5x3 binary columns. Votes only: target-agnostic."""
    return np.concatenate([np.eye(3)[V[nm][rows]] for nm, _, _ in READERS], axis=1)


def main():
    tgt, gold, V = build()
    targets = ['Covid Vaccine', 'Digital Transformation', 'Women empowerment']
    print(f"pool = {len(gold)} labelled rows over {len(targets)} targets; "
          f"{len(READERS)} readers -> {3*len(READERS)} features\n")
    print(f"{'held-out target':26s} {'best single':>22s} {'majority':>10s} {'STACKER':>10s}")
    agg = {}
    for T in targets:
        te = np.where(tgt == T)[0]
        trn = np.where(tgt != T)[0]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(feats(V, trn), gold[trn])
        pred = clf.predict(feats(V, te))
        st = S.compute_metrics(gold[te], pred)['Favg2'] * 100
        singles = {nm: S.compute_metrics(gold[te], V[nm][te])['Favg2'] * 100
                   for nm, _, _ in READERS}
        bn = max(singles, key=singles.get)
        mj = []
        for i in te:
            v = [V[nm][i] for nm, _, _ in READERS]
            mj.append(max(set(v), key=v.count))
        mjs = S.compute_metrics(gold[te], np.array(mj))['Favg2'] * 100
        print(f"  {T:24s} {bn:>12s} {singles[bn]:6.2f} {mjs:10.2f} {st:10.2f}"
              f"   {'WIN' if st > singles[bn] else 'lose'} vs best-single ({st-singles[bn]:+.2f})")
        agg[T] = dict(stacker=st, best_single=singles[bn], best_name=bn, majority=mjs,
                      singles={k: round(v, 2) for k, v in singles.items()})
    m = lambda k: np.mean([agg[T][k] for T in targets])
    print(f"\n  MEAN over 3 unseen targets: best-single {m('best_single'):.2f} · "
          f"majority {m('majority'):.2f} · STACKER {m('stacker'):.2f}")
    json.dump(agg, open('_llm/stacker_loto.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
