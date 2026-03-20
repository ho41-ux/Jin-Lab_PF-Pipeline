from __future__ import annotations

from functools import lru_cache
from io import StringIO
import re
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

INVENTORY_SEARCH_URL = "https://hfpappexternal.fda.gov/scripts/fdcc/index.cfm"
REQUEST_TIMEOUT = 15
MAX_RESULTS = 25


def _clean(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


@lru_cache(maxsize=256)
def fetch_gras_html(query: str) -> str:
    params = {
        "set": "grasnotices",
        "sort": "GRN_No",
        "order": "DESC",
        "showAll": "true",
        "type": "basic",
        "search": query,
    }
    resp = requests.get(INVENTORY_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_clean(c).lower() for c in out.columns]
    return out


DATE_RE = r"[A-Z][a-z]{2} \d{1,2}, \d{4}"
RESPONSE_RE = r"(?:FDA has no questions[^\n]*|At the notifier's request, FDA ceased to evaluate this notice[^\n]*)"
INLINE_RE = re.compile(
    rf"^(?P<substance>.+?)\s+(?P<date>{DATE_RE})\s+(?P<response>{RESPONSE_RE})(?:\s+\([^\)]*\))?(?:\s+{DATE_RE})?$"
)


def _parse_text_fallback(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    lines = [_clean(x) for x in text.splitlines()]
    lines = [x for x in lines if x]
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def next_grn(start: int) -> str:
        for j in range(start, min(start + 5, len(lines))):
            if re.fullmatch(r"\d{3,4}", lines[j]):
                return lines[j]
        return ""

    for i, line in enumerate(lines):
        candidate = line
        m = INLINE_RE.match(candidate)
        consumed = 0
        if not m and i + 1 < len(lines):
            candidate = f"{line} {lines[i + 1]}"
            m = INLINE_RE.match(candidate)
            consumed = 1
        if not m:
            continue
        grn = next_grn(i + 1 + consumed)
        substance = _clean(m.group("substance"))
        date = _clean(m.group("date"))
        response = _clean(m.group("response"))
        key = (grn, substance)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "grn": grn,
                "substance": substance,
                "inventory_date": date,
                "response": response,
                "notice_url": "",
            }
        )
        if len(results) >= MAX_RESULTS:
            break
    return results


def _parse_table_records(html: str) -> list[dict[str, str]]:
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        tables = []

    records: list[dict[str, str]] = []
    for raw in tables:
        df = _normalize_columns(raw)
        cols = set(df.columns)
        if not any("grn" in c for c in cols):
            continue
        if not any("substance" in c or "name" in c for c in cols):
            continue

        grn_col = next((c for c in df.columns if "grn" in c), None)
        substance_col = next((c for c in df.columns if "substance" in c or "name" in c), None)
        date_col = next((c for c in df.columns if "date" in c), None)
        response_col = next((c for c in df.columns if "response" in c or "letter" in c or "status" in c), None)
        link_col = next((c for c in df.columns if "link" in c or "pdf" in c or "url" in c), None)

        if not grn_col or not substance_col:
            continue

        for _, row in df.iterrows():
            grn = _clean(row.get(grn_col, ""))
            substance = _clean(row.get(substance_col, ""))
            if not grn and not substance:
                continue
            if not re.search(r"\d{3,4}", grn):
                continue
            records.append(
                {
                    "grn": re.search(r"\d{3,4}", grn).group(0),
                    "substance": substance,
                    "inventory_date": _clean(row.get(date_col, "")) if date_col else "",
                    "response": _clean(row.get(response_col, "")) if response_col else "",
                    "notice_url": _clean(row.get(link_col, "")) if link_col else "",
                }
            )
    return records


def _parse_notice_links(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    links: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        text = _clean(a.get_text(" ", strip=True))
        href = urljoin(INVENTORY_SEARCH_URL, a["href"])
        if "fda.gov" not in href:
            continue
        if "PDF" in text or "in PDF" in text or href.lower().endswith(".pdf"):
            # nearest numeric token in surrounding text is often the GRN, but we can't rely on DOM.
            continue
    return links


@lru_cache(maxsize=256)
def search_gras_inventory(query: str) -> list[dict[str, str]]:
    q = _clean(query)
    if not q:
        return []
    try:
        html = fetch_gras_html(q)
    except Exception:
        return []

    records = _parse_table_records(html)
    if not records:
        records = _parse_text_fallback(html)

    # Post-filter to improve precision.
    terms = [t for t in re.split(r"\s+", q.lower()) if len(t) > 2]
    filtered: list[dict[str, str]] = []
    for rec in records:
        hay = f"{rec.get('substance','')} {rec.get('response','')}".lower()
        if all(term in hay for term in terms[:2]) or any(term in hay for term in terms):
            filtered.append(rec)
    if filtered:
        records = filtered

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        key = (_clean(rec.get("grn")), _clean(rec.get("substance")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
        if len(deduped) >= MAX_RESULTS:
            break
    return deduped


def summarize_gras_hits(query: str) -> dict[str, Any]:
    hits = search_gras_inventory(query)
    return {
        "query": query,
        "has_hit": bool(hits),
        "grns": sorted({h.get("grn", "") for h in hits if h.get("grn")}),
        "hits": hits,
    }
