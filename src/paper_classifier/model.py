"""TF-IDF + logistic regression classifier for empirical vs. non-empirical papers."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .features import SIGNAL_NAMES, meta_text, paper_text, signal_matrix

FT_NAMES = SIGNAL_NAMES + ["_tables"]


class EmpiricalClassifier:
    """mode='baseline': TF-IDF on title+abstract only.
    mode='enhanced': + venue/keyword TF-IDF, statistical-signal counts, missing-abstract flag,
    and full-text signal counts when a PDF was readable."""

    def __init__(self, mode: str = "enhanced", C: float = 4.0):
        self.mode = mode
        self.text_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english")
        self.meta_vec = TfidfVectorizer(token_pattern=r"[^;]+", min_df=2, lowercase=True)
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)

    def _dense(self, recs: list[dict]) -> np.ndarray:
        sig = signal_matrix([paper_text(r) for r in recs])
        no_abs = np.array([[0.0 if r.get("abstract") else 1.0] for r in recs])
        ft = np.zeros((len(recs), len(FT_NAMES) + 1))
        for i, r in enumerate(recs):
            if r.get("fulltext"):
                ft[i, :-1] = np.log1p([r["fulltext"].get(k, 0) for k in FT_NAMES])
                ft[i, -1] = 1.0
        return np.hstack([sig, no_abs, ft])

    def _X(self, recs: list[dict], fit: bool = False):
        texts = [paper_text(r) for r in recs]
        Xt = self.text_vec.fit_transform(texts) if fit else self.text_vec.transform(texts)
        if self.mode == "baseline":
            return Xt
        metas = [meta_text(r) or "none" for r in recs]
        Xm = self.meta_vec.fit_transform(metas) if fit else self.meta_vec.transform(metas)
        return hstack([Xt, Xm, csr_matrix(self._dense(recs))]).tocsr()

    def fit(self, recs: list[dict], y) -> "EmpiricalClassifier":
        self.clf.fit(self._X(recs, fit=True), np.asarray(y))
        return self

    def predict_proba(self, recs: list[dict]) -> np.ndarray:
        return self.clf.predict_proba(self._X(recs))[:, 1]


def scores(y_true, prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
