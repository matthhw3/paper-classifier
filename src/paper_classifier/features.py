"""Text building, statistical-signal features, and optional in-memory full-text reading."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import requests

# Signals that a paper runs statistical analysis (the "empirical" criterion).
STAT_PATTERNS = {
    "regression": r"\bregress(?:ion|ions|ed|ing)?\b",
    "estimate": r"\bestimat(?:e|es|ed|ion|ing)\b",
    "significance": r"\bstatistically significant|\bsignifican(?:t|ce) at\b|\bp\s*[<=]\s*0?\.\d",
    "std_error": r"\bstandard errors?\b|\bt-stat",
    "sample": r"\bsample of\b|\b\d[\d,]*\s+(?:firms|firm-years|observations|respondents|participants|employees|managers|subjects)\b",
    "data": r"\b(?:panel|survey|archival|field|experimental|administrative) data\b|\bdataset\b|\bdata set\b",
    "experiment": r"\b(?:field|laboratory|lab|natural|randomi[sz]ed) experiment|\brandomi[sz]ed\b",
    "identification": r"\bdifference-in-differences\b|\binstrumental variables?\b|\bfixed effects\b|\bpropensity score\b|\bstructural estimation\b",
    "tests": r"\bwe (?:test|find|document|show empirically)\b|\bhypothes[ie]s\b",
    "evidence": r"\bempirical(?:ly)?\b|\bevidence\b",
}
# Signals of non-statistical work (theory, case study, qualitative, commentary).
NONSTAT_PATTERNS = {
    "theory": r"\bwe (?:develop|propose|present) a (?:model|theory|framework)\b|\btheorem\b|\bproposition\b|\bequilibri(?:um|a)\b|\bwe prove\b",
    "case_study": r"\bcase stud(?:y|ies)\b|\bethnograph|\binterviews?\b|\bqualitative\b|\bfield ?work\b",
    "commentary": r"\bcommentary\b|\bessay\b|\breview(?:s|ing)? (?:the|of) literature\b|\bdiscussion\b|\bbook review\b|\bcomment on\b|\breply\b",
}
ALL_PATTERNS = {**{f"s_{k}": v for k, v in STAT_PATTERNS.items()}, **{f"n_{k}": v for k, v in NONSTAT_PATTERNS.items()}}
_COMPILED = {k: re.compile(v, re.I) for k, v in ALL_PATTERNS.items()}
SIGNAL_NAMES = list(ALL_PATTERNS)


def signal_counts(text: str) -> dict[str, int]:
    return {k: len(p.findall(text or "")) for k, p in _COMPILED.items()}


def paper_text(rec: dict) -> str:
    return " ".join(filter(None, [rec.get("title"), rec.get("abstract")]))


def meta_text(rec: dict) -> str:
    return " ; ".join(filter(None, [rec.get("venue"), rec.get("type"), rec.get("keywords"), rec.get("topics")]))


def signal_matrix(texts: list[str]) -> np.ndarray:
    m = np.array([[c for c in signal_counts(t).values()] for t in texts], dtype=float)
    return np.log1p(m)


class FullText:
    """Fetch open-access PDFs into memory and keep only signal counts (no files saved)."""

    def __init__(self, cache_dir: str | Path = ".cache", max_mb: int = 25, max_pages: int = 40):
        self.path = Path(cache_dir) / "fulltext_signals.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.max_bytes = max_mb * 1024 * 1024
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (paper-classifier research tool)"

    def save(self) -> None:
        self.path.write_text(json.dumps(self.cache))

    def signals(self, url: str | None) -> dict[str, int] | None:
        if not url:
            return None
        if url not in self.cache:
            self.cache[url] = self._fetch(url)
        return self.cache[url]

    def _fetch(self, url: str) -> dict[str, int] | None:
        from pypdf import PdfReader

        try:
            r = self.session.get(url, timeout=30, stream=True)
            if r.status_code != 200 or "pdf" not in r.headers.get("Content-Type", "").lower():
                return None
            buf = io.BytesIO()
            for chunk in r.iter_content(1 << 16):
                buf.write(chunk)
                if buf.tell() > self.max_bytes:
                    return None
            reader = PdfReader(buf)
            text = " ".join((p.extract_text() or "") for p in reader.pages[: self.max_pages])
            counts = signal_counts(text)
            counts["_tables"] = len(re.findall(r"\bTable \d", text))
            return counts
        except Exception:
            return None
