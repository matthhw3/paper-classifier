"""Resolve links to metadata (title, abstract, venue, topics, OA PDF) via OpenAlex."""

from __future__ import annotations

import difflib
import json
import os
import time
from pathlib import Path

import requests

API = "https://api.openalex.org/works"
FIELDS = "id,doi,title,abstract_inverted_index,primary_location,keywords,topics,type,best_oa_location"
BATCH = 50


class Resolver:
    def __init__(self, cache_dir: str | Path = ".cache", mailto: str | None = None):
        self.cache_path = Path(cache_dir) / "openalex.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.session = requests.Session()
        self.params = {"mailto": mailto or os.environ.get("OPENALEX_MAILTO", "")}

    def save(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache))

    def _get(self, params: dict) -> list[dict]:
        for attempt in range(4):
            r = self.session.get(API, params={**self.params, **params}, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json().get("results", [])
        return []

    def _batch(self, key: str, ids: list[str]) -> None:
        todo = [i for i in dict.fromkeys(ids) if f"{key}:{i}" not in self.cache]
        for start in range(0, len(todo), BATCH):
            chunk = todo[start : start + BATCH]
            results = self._get({"filter": f"{key}:" + "|".join(chunk), "per-page": BATCH, "select": FIELDS})
            found = {}
            for w in results:
                rec = _simplify(w)
                found[rec["doi"] if key == "doi" else rec["openalex"]] = rec
            for i in chunk:
                self.cache[f"{key}:{i}"] = found.get(i)
            time.sleep(0.1)
        self.save()

    def _by_title(self, title: str) -> dict | None:
        k = f"title:{title.lower()}"
        if k not in self.cache:
            results = self._get({"search": title, "per-page": 3, "select": FIELDS})
            best = None
            for w in results:
                ratio = difflib.SequenceMatcher(None, title.lower(), (w.get("title") or "").lower()).ratio()
                if ratio >= 0.9:
                    best = _simplify(w)
                    break
            self.cache[k] = best
            time.sleep(0.1)
        return self.cache[k]

    def resolve_all(self, ids: list[dict], titles: list[str]) -> list[dict | None]:
        self._batch("doi", [i["doi"] for i in ids if i["doi"]])
        self._batch("openalex", [i["openalex"] for i in ids if i["openalex"] and not i["doi"]])
        out = []
        for i, title in zip(ids, titles):
            rec = None
            if i["doi"]:
                rec = self.cache.get(f"doi:{i['doi']}")
            if rec is None and i["openalex"]:
                rec = self.cache.get(f"openalex:{i['openalex']}")
            if rec is None and title:
                rec = self._by_title(title)
            out.append(rec)
        self.save()
        return out


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {p: w for w, ps in inv.items() for p in ps}
    return " ".join(pos[i] for i in sorted(pos))


def _simplify(w: dict) -> dict:
    loc = w.get("primary_location") or {}
    oa = w.get("best_oa_location") or {}
    return {
        "openalex": (w.get("id") or "").rsplit("/", 1)[-1],
        "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower() or None,
        "title": w.get("title") or "",
        "abstract": _abstract(w.get("abstract_inverted_index")),
        "venue": ((loc.get("source") or {}).get("display_name")) or "",
        "keywords": "; ".join(k.get("display_name", "") for k in w.get("keywords") or []),
        "topics": "; ".join(t.get("display_name", "") for t in w.get("topics") or []),
        "type": w.get("type") or "",
        "pdf_url": oa.get("pdf_url"),
    }
