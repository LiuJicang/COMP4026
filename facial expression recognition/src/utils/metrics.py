import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def expression_consistency_rate(pred_orig, pred_anon):
    pred_orig = np.asarray(pred_orig)
    pred_anon = np.asarray(pred_anon)
    if len(pred_orig) == 0:
        return 0.0
    return float((pred_orig == pred_anon).mean())
