"""Load a spreadsheet of paper links and pull out identifiers + optional labels."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)
OPENALEX_RE = re.compile(r"openalex\.org/(W\d+)", re.I)

LINK_NAMES = ("url", "link", "links", "doi", "paper", "paper_url")
TRUE_VALUES = {"true", "yes", "y", "1", "empirical"}
FALSE_VALUES = {"false", "no", "n", "0", "not empirical"}


def _read_raw(path: Path, header) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=header, dtype=str)
    return pd.read_csv(path, header=header, dtype=str, encoding_errors="replace")


def load_sheet(path: str | Path) -> pd.DataFrame:
    """Read a CSV/XLSX, skipping decorative group-header rows above the real header."""
    path = Path(path)
    preview = _read_raw(path, header=None).head(5)
    header_row = 0
    for i, row in preview.iterrows():
        cells = {str(c).strip().lower() for c in row if pd.notna(c)}
        if cells & set(LINK_NAMES):
            header_row = i
            break
    df = _read_raw(path, header=header_row).fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_link_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred:
        if preferred not in df.columns:
            raise KeyError(f"column {preferred!r} not in sheet")
        return preferred
    for name in LINK_NAMES:
        for col in df.columns:
            if col.lower() == name:
                return col
    # Fall back to whichever column looks most like URLs.
    best, best_share = None, 0.0
    for col in df.columns:
        share = df[col].astype(str).str.startswith("http").mean()
        if share > best_share:
            best, best_share = col, share
    if best is None or best_share < 0.5:
        raise ValueError("could not find a column of links; pass --link-col")
    return best


def extract_ids(link: str, doi_hint: str = "") -> dict:
    """Return {'doi': ..., 'openalex': ...} for a link (either may be None)."""
    out = {"doi": None, "openalex": None}
    for text in (doi_hint, link):
        m = DOI_RE.search(text or "")
        if m:
            out["doi"] = m.group(0).rstrip(".,;)").lower()
            break
    m = OPENALEX_RE.search(link or "")
    if m:
        out["openalex"] = m.group(1)
    return out


def parse_label(value: str) -> int | None:
    v = str(value).strip().lower()
    if v in TRUE_VALUES:
        return 1
    if v in FALSE_VALUES:
        return 0
    return None  # "No Access", "Duplicate", blank, etc. -> unlabeled
