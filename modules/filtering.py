from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_hosts(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_safety_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def detect_hosts_in_text(text: str, hosts_df: pd.DataFrame) -> list[str]:
    matches: list[str] = []
    low = text.lower()
    for _, row in hosts_df.iterrows():
        aliases = [x.strip() for x in str(row["aliases"]).split(";") if x.strip()]
        all_terms = [str(row["display_name"]).strip(), *aliases]
        if any(_contains_term(low, term) for term in all_terms):
            if bool(row["allowed"]):
                matches.append(str(row["display_name"]))
    return sorted(set(matches))


def detect_organisms_in_text(text: str, safety_df: pd.DataFrame) -> list[dict]:
    matches: list[dict] = []
    low = text.lower()
    for _, row in safety_df.iterrows():
        aliases = [x.strip() for x in str(row["aliases"]).split(";") if x.strip()]
        all_terms = [str(row["organism_name"]).strip(), *aliases]
        if any(_contains_term(low, term) for term in all_terms):
            matches.append(row.to_dict())
    return matches


def score_record(allowed_host_match: bool, excluded_count: int, text: str) -> int:
    score = 0
    if allowed_host_match:
        score += 5

    low = text.lower()
    bonus_terms = ["heterologous", "engineered", "biosynthesis", "pathway", "overexpression"]
    score += sum(1 for term in bonus_terms if term in low)
    score -= 3 * excluded_count
    return score
