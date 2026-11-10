"""Command-line entry point.

  paper-classifier run SHEET --sample 200 --oa-only --fulltext
      Resolve every link, pick a random sample, read open-access full text for it (throttled),
      and score abstract-only vs. enhanced models against the sheet's human labels.

  paper-classifier classify NEW_SHEET --train LABELED_SHEET
      Label a new spreadsheet of links using a model trained on a labeled one.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .features import FullText
from .io import extract_ids, find_link_column, load_sheet, parse_label
from .model import EmpiricalClassifier, scores
from .resolve import Resolver

REVIEW_BAND = (0.35, 0.65)


def build_records(df: pd.DataFrame, link_col: str, resolver: Resolver) -> list[dict]:
    """OpenAlex metadata per row, falling back to any title/abstract already in the sheet."""
    doi_col = next((c for c in df.columns if c.lower() == "doi"), None)
    ids = [extract_ids(link, df[doi_col].iloc[i] if doi_col else "") for i, link in enumerate(df[link_col])]
    titles = list(df["title"]) if "title" in df.columns else [""] * len(df)
    found = resolver.resolve_all(ids, titles)
    recs = []
    for i, rec in enumerate(found):
        rec = dict(rec or {})
        rec["link"] = df[link_col].iloc[i]
        rec["resolved"] = bool(found[i])
        for field in ("title", "abstract"):
            if not rec.get(field) and field in df.columns:
                rec[field] = df[field].iloc[i]
        recs.append(rec)
    return recs


def add_fulltext(recs: list[dict], idx: list[int], args) -> None:
    ft = FullText(delay=args.delay, limit=args.limit, publisher=args.publisher_pdfs)
    for n, i in enumerate(idx, 1):
        recs[i]["fulltext"] = ft.signals_for(recs[i])
        if n % 10 == 0 or n == len(idx):
            got = sum(bool(recs[j].get("fulltext")) for j in idx[:n])
            print(f"  full text: {n}/{len(idx)} checked, {got} readable, {ft.remaining} requests left", flush=True)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _keys(rec: dict) -> set[str]:
    """Identity keys used to spot the same paper appearing twice."""
    keys = set()
    if rec.get("doi"):
        keys.add("doi:" + rec["doi"].lower())
    if len(_norm(rec.get("title"))) > 15:
        keys.add("title:" + _norm(rec.get("title"))[:120])
    if len(_norm(rec.get("abstract"))) > 200:
        keys.add("abs:" + _norm(rec.get("abstract"))[:300])
    return keys


def duplicate_groups(recs: list[dict]) -> np.ndarray:
    """Union papers sharing any identity key into one group id."""
    parent = list(range(len(recs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[str, int] = {}
    for i, r in enumerate(recs):
        for k in _keys(r):
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    return np.array([find(i) for i in range(len(recs))])


def assert_no_overlap(recs: list[dict], train_idx, test_idx) -> None:
    train_keys = set().union(*(_keys(recs[i]) for i in train_idx))
    leaks = [i for i in test_idx if _keys(recs[i]) & train_keys]
    if leaks:
        raise RuntimeError(f"contamination: {len(leaks)} test papers also appear in training")


def _without_fulltext(recs: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "fulltext"} for r in recs]


def cmd_run(args) -> None:
    df = load_sheet(args.sheet)
    link_col = find_link_column(df, args.link_col)
    print(f"{len(df)} rows, links in column {link_col!r}")
    recs = build_records(df, link_col, Resolver())
    labels = np.array([parse_label(v) if args.label_col in df.columns else None for v in df.get(args.label_col, [""] * len(df))], dtype=object)
    print(f"resolved {sum(r['resolved'] for r in recs)}/{len(recs)}; labeled {sum(l is not None for l in labels)}")

    pool = [i for i, r in enumerate(recs) if r.get("pdf_urls")] if args.oa_only else list(range(len(recs)))
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(pool, min(args.sample, len(pool))))
    print(f"sample: {len(sample)} papers ({'open-access only' if args.oa_only else 'all'}; pool {len(pool)})")

    if args.fulltext:
        add_fulltext(recs, sample, args)

    # Cross-validate on the labeled sample; every fold also trains on all labeled papers outside the sample.
    # Duplicates (same DOI/title/abstract) share a group and never straddle train and test.
    groups = duplicate_groups(recs)
    test = np.array([i for i in sample if labels[i] is not None])
    sample_groups = set(groups[test])
    outside = [i for i in range(len(recs)) if labels[i] is not None and groups[i] not in sample_groups]
    n_dropped = sum(1 for i in range(len(recs)) if labels[i] is not None and i not in set(sample) and groups[i] in sample_groups)
    print(f"training pool outside sample: {len(outside)} labeled papers ({n_dropped} dropped as duplicates of sample papers)")
    y = labels.astype(object)
    probs = {m: np.zeros(len(test)) for m in ("baseline", "enhanced_no_fulltext", "enhanced")}
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (tr, te) in enumerate(skf.split(test, y[test].astype(int), groups[test]), 1):
        train_idx = outside + list(test[tr])
        assert_no_overlap(recs, train_idx, test[te])
        print(f"  fold {fold}: train {len(train_idx)}, test {len(te)}, overlap check passed", flush=True)
        ytr = y[train_idx].astype(int)
        tr_recs = [recs[i] for i in train_idx]
        te_recs = [recs[i] for i in test[te]]
        probs["baseline"][te] = EmpiricalClassifier("baseline").fit(tr_recs, ytr).predict_proba(te_recs)
        probs["enhanced_no_fulltext"][te] = (
            EmpiricalClassifier("enhanced").fit(_without_fulltext(tr_recs), ytr).predict_proba(_without_fulltext(te_recs)))
        probs["enhanced"][te] = EmpiricalClassifier("enhanced").fit(tr_recs, ytr).predict_proba(te_recs)

    yt = y[test].astype(int)
    print(f"\nscored on {len(test)} labeled sample papers (5-fold CV):")
    print(f"  {'model':<22}{'accuracy':>9}{'precision':>10}{'recall':>8}{'f1':>6}")
    for m, p in probs.items():
        s = scores(yt, p)
        print(f"  {m:<22}{s['accuracy']:>9.1%}{s['precision']:>10.1%}{s['recall']:>8.1%}{s['f1']:>6.2f}")

    p = probs["enhanced"]
    out = pd.DataFrame({
        "link": [recs[i]["link"] for i in test],
        "title": [recs[i].get("title", "") for i in test],
        "predicted_empirical": p >= 0.5,
        "probability": p.round(3),
        "needs_review": (p > REVIEW_BAND[0]) & (p < REVIEW_BAND[1]),
        "fulltext_read": [bool(recs[i].get("fulltext")) for i in test],
        "human_label": yt.astype(bool),
    })
    out["correct"] = out["predicted_empirical"] == out["human_label"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({out['needs_review'].sum()} flagged for human review)")


def cmd_classify(args) -> None:
    resolver = Resolver()
    train_df = load_sheet(args.train)
    train_recs = build_records(train_df, find_link_column(train_df, args.link_col), resolver)
    y = [parse_label(v) for v in train_df[args.label_col]]
    keep = [i for i, v in enumerate(y) if v is not None]
    model = EmpiricalClassifier("enhanced").fit([train_recs[i] for i in keep], [y[i] for i in keep])

    df = load_sheet(args.sheet)
    recs = build_records(df, find_link_column(df, args.link_col), resolver)
    p = model.predict_proba(recs)
    df["predicted_empirical"] = p >= 0.5
    df["probability"] = p.round(3)
    df["needs_review"] = (p > REVIEW_BAND[0]) & (p < REVIEW_BAND[1])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {int((p >= 0.5).sum())}/{len(df)} predicted empirical, {int(df['needs_review'].sum())} need review")


def main() -> None:
    ap = argparse.ArgumentParser(prog="paper-classifier", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="evaluate on a random sample of a labeled sheet")
    run.add_argument("sheet")
    run.add_argument("--sample", type=int, default=200)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--oa-only", action="store_true", help="sample only papers with an open-access PDF")
    run.add_argument("--fulltext", action="store_true", help="read sampled PDFs in memory (throttled)")
    run.add_argument("--out", default="output/sample_results.csv")
    run.set_defaults(func=cmd_run)

    cls = sub.add_parser("classify", help="label a new sheet of links")
    cls.add_argument("sheet")
    cls.add_argument("--train", required=True, help="labeled sheet to train on")
    cls.add_argument("--out", default="output/classified.csv")
    cls.set_defaults(func=cmd_classify)

    for p in (run, cls):
        p.add_argument("--link-col")
        p.add_argument("--label-col", default="empirical")
        p.add_argument("--delay", type=float, default=15.0, help="seconds between PDF requests")
        p.add_argument("--limit", type=int, default=400, help="max PDF requests per run")
        p.add_argument("--publisher-pdfs", action="store_true", help="also try publisher PDF links (only where permitted)")

    args = ap.parse_args()
    args.func(args)
