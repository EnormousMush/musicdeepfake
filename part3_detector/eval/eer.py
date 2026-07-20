"""Equal Error Rate — the project's primary metric."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> dict:
    """EER + threshold + AUC. labels: 1=AI, 0=human; scores: P(AI)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return {
        "eer": eer,
        "threshold": float(thr[idx]),
        "auc": float(roc_auc_score(labels, scores)),
    }
