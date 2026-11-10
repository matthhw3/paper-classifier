"""Robustness experiments for the empirical-paper classifier (no network needed; uses caches).

  uv run python scripts/experiments.py SHEET

1. Paired test: abstract-only vs. full-text model on the same open-access sample.
2. Abstract-only accuracy on ALL labeled papers, broken down by subgroup.
3. Learning curve: how accuracy changes with the number of training papers.
4. Weaker setups: title only, simpler TF-IDF, single random train/test splits.
"""

from __future__ import annotations

import random
import sys
from math import comb

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from paper_classifier.cli import build_records, duplicate_groups
from paper_classifier.features import FullText
from paper_classifier.io import find_link_column, load_sheet, parse_label
from paper_classifier.model import EmpiricalClassifier
from paper_classifier.resolve import Resolver

SEED = 42


def mcnemar_exact(a_right: np.ndarray, b_right: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired correct/incorrect outcomes."""
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    n, k = b + c, min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return b, c, p


def cv_predict(recs, y, groups, idx, make_model, train_extra=(), folds=5, seed=SEED):
    """Grouped stratified CV over idx; returns out-of-fold probabilities aligned with idx."""
    idx = np.asarray(idx)
    out = np.zeros(len(idx))
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(idx, y[idx], groups[idx]):
        tr_idx = list(train_extra) + list(idx[tr])
        m = make_model().fit([recs[i] for i in tr_idx], y[tr_idx])
        out[te] = m.predict_proba([recs[i] for i in idx[te]])
    return out


class SimpleBaseline:
    """A plainer setup: unigram counts, no class weighting, default regularization."""

    def __init__(self, field="both"):
        self.field = field
        self.vec = CountVectorizer(min_df=1)
        self.clf = LogisticRegression(max_iter=5000)

    def _t(self, recs):
        if self.field == "title":
            return [r.get("title") or "" for r in recs]
        return [f"{r.get('title') or ''} {r.get('abstract') or ''}" for r in recs]

    def fit(self, recs, y):
        self.clf.fit(self.vec.fit_transform(self._t(recs)), y)
        return self

    def predict_proba(self, recs):
        return self.clf.predict_proba(self.vec.transform(self._t(recs)))[:, 1]


class TitleOnly(EmpiricalClassifier):
    def __init__(self):
        super().__init__("baseline")

    def _X(self, recs, fit=False):
        return super()._X([{"title": r.get("title")} for r in recs], fit)


def acc(y, p):
    return float(np.mean((p >= 0.5).astype(int) == y))


def main(path: str) -> None:
    df = load_sheet(path)
    recs = build_records(df, find_link_column(df), Resolver())
    raw = [parse_label(v) for v in df["empirical"]]
    labeled = np.array([i for i, v in enumerate(raw) if v is not None])
    y = np.array([v if v is not None else -1 for v in raw])
    groups = duplicate_groups(recs)
    has_abs = np.array([len((r.get("abstract") or "").split()) >= 30 for r in recs])
    is_oa = np.array([bool(r.get("pdf_urls")) for r in recs])

    # Recreate the open-access sample exactly as `run` does and attach cached full-text signals.
    pool = [i for i, r in enumerate(recs) if r.get("pdf_urls")]
    sample = sorted(random.Random(SEED).sample(pool, min(200, len(pool))))
    ft = FullText(limit=0)  # limit=0: cache only, no network
    for i in sample:
        recs[i]["fulltext"] = ft.signals_for(recs[i])
    test = np.array([i for i in sample if raw[i] is not None])
    sg = set(groups[test])
    outside = [i for i in labeled if groups[i] not in sg]

    print("=" * 70, "\n1. Paired test on the open-access sample (same papers, same folds)")
    p_base = cv_predict(recs, y, groups, test, lambda: EmpiricalClassifier("baseline"), outside)
    p_full = cv_predict(recs, y, groups, test, lambda: EmpiricalClassifier("enhanced"), outside)
    rb, rf = (p_base >= 0.5) == y[test], (p_full >= 0.5) == y[test]
    b, c, p = mcnemar_exact(rb, rf)
    print(f"  baseline {rb.mean():.1%} vs full-text {rf.mean():.1%} on {len(test)} papers")
    print(f"  only baseline right: {b}, only full-text right: {c}, exact McNemar p = {p:.2f}")

    print("=" * 70, "\n2. Abstract-only model on ALL labeled papers (5-fold grouped CV)")
    p_all = cv_predict(recs, y, groups, labeled, lambda: EmpiricalClassifier("baseline"))
    right = (p_all >= 0.5) == y[labeled]
    print(f"  all labeled: {right.mean():.1%} (n={len(labeled)}, {np.mean(y[labeled]):.0%} empirical)")
    for name, mask in [("open-access", is_oa), ("not open-access", ~is_oa),
                       ("has abstract (30+ words)", has_abs), ("no/short abstract", ~has_abs)]:
        m = mask[labeled]
        print(f"  {name:<26} {right[m].mean():.1%}  (n={m.sum()})")
    venues = np.array([df["journal_input"].iloc[i].replace("The ", "") for i in labeled])
    print("  by journal (n>=40):")
    for v in sorted(set(venues), key=lambda v: -np.sum(venues == v)):
        m = venues == v
        if m.sum() >= 40:
            print(f"    {v[:38]:<40}{right[m].mean():.1%}  (n={m.sum()}, {np.mean(y[labeled][m]):.0%} empirical)")

    print("=" * 70, "\n3. Learning curve (abstract-only; fixed 20% test folds, 5 repeats)")
    rng = np.random.default_rng(SEED)
    sizes = [50, 100, 200, 400, 600, 800, None]
    res = {s: [] for s in sizes}
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(labeled, y[labeled], groups[labeled]):
        tr_all, te_idx = labeled[tr], labeled[te]
        for s in sizes:
            for _ in range(1 if s is None else 3):
                sub = tr_all if s is None else rng.choice(tr_all, s, replace=False)
                m = EmpiricalClassifier("baseline").fit([recs[i] for i in sub], y[sub])
                res[s].append(acc(y[te_idx], m.predict_proba([recs[i] for i in te_idx])))
    for s in sizes:
        print(f"  train on {('all ~' + str(len(labeled) * 4 // 5)) if s is None else s:>8} papers: {np.mean(res[s]):.1%}  (min {min(res[s]):.1%}, max {max(res[s]):.1%})")

    print("=" * 70, "\n4. Weaker setups (5-fold grouped CV on all labeled)")
    for name, make in [("title only", TitleOnly),
                       ("simple counts, no class weights", SimpleBaseline),
                       ("simple counts, title only", lambda: SimpleBaseline("title"))]:
        print(f"  {name:<34} {acc(y[labeled], cv_predict(recs, y, groups, labeled, make)):.1%}")

    print("  single 80/20 split with only 600 papers (how much one split varies by luck):")
    accs = []
    for seed in range(30):
        sub = np.random.default_rng(seed).choice(labeled, 600, replace=False)
        tr, te = train_test_split(sub, test_size=0.2, stratify=y[sub], random_state=seed)
        accs.append(acc(y[te], SimpleBaseline().fit([recs[i] for i in tr], y[tr]).predict_proba([recs[i] for i in te])))
    print(f"    simple model over 30 random splits: mean {np.mean(accs):.1%}, range {min(accs):.1%} to {max(accs):.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
