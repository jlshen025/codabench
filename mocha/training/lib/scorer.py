"""Metrics matching the CARE-PD / MoCha2026 leaderboard.

PRIMARY: macro-F1 over classes {0,1,2} that appear in the ground truth
(CARE-PD const LABELS_INCLUDED_IN_F1_CALCULATION=[0,1,2]; class 3 excluded from
the F1 average but NOT from the confusion / other metrics). We also report the
all-4-class macro-F1 in case MoCha's hidden scorer averages over every class
present — disambiguated empirically by the baseline_v0 server score.
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score

F1_LABELS = [0, 1, 2]  # CARE-PD convention


def macro_f1_012(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = [l for l in F1_LABELS if l in set(y_true.tolist())]
    if not labels:
        return 0.0
    return float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0))


def macro_f1_all(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()))
    return float(f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0))


def macro_f1_4fixed(y_true, y_pred):
    """SERVER-FAITHFUL: macro-F1 over the fixed 4 labels [0,1,2,3] (÷4 always;
    a class with no support contributes F1=0). The MoCha hidden scorer divides
    by 4 (proven by the constant=1 baseline), and the hidden test contains
    class 3 — so this is the metric to SELECT on."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(f1_score(y_true, y_pred, average='macro', labels=[0, 1, 2, 3], zero_division=0))


def qwk(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=[0, 1, 2, 3]))
    except Exception:
        return 0.0


def all_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels012 = [l for l in F1_LABELS if l in set(y_true.tolist())]
    return {
        'macro_f1_4fixed': macro_f1_4fixed(y_true, y_pred),
        'macro_f1_012': macro_f1_012(y_true, y_pred),
        'macro_f1_all': macro_f1_all(y_true, y_pred),
        'macro_prec_012': float(precision_score(y_true, y_pred, average='macro', labels=labels012, zero_division=0)) if labels012 else 0.0,
        'macro_rec_012': float(recall_score(y_true, y_pred, average='macro', labels=labels012, zero_division=0)) if labels012 else 0.0,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'qwk': qwk(y_true, y_pred),
        'n': int(len(y_true)),
    }
