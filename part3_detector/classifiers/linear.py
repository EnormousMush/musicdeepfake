"""
Linear-probe classifier: standardize features + logistic regression.
The cheap backend used to rank encoders/layers before investing in AASIST.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

NAME = "logreg"


def train(X: np.ndarray, y: np.ndarray, cfg: dict):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=cfg.get("C", 1.0), max_iter=2000, class_weight="balanced"),
    )
    clf.fit(X, y)
    return clf


def score(clf, X: np.ndarray) -> np.ndarray:
    """Return P(label == AI) for each row."""
    return clf.predict_proba(X)[:, 1]
