from __future__ import annotations

import html
import re
from typing import Iterable

import pandas as pd

# Conservative mode: keep only tokens that strongly look like genes or are on a curated list.
GENE_EXACT_ALLOW = {
    # carotenoid / apocarotenoid
    "crtE", "crtB", "crtI", "crtY", "crtYB", "crtZ", "crtW", "crtL", "crtR", "crtO", "crtU",
    "carB", "carRA", "carRP", "carRPB", "carX", "carO", "carS",
    "blh", "BKT", "bkt", "chyB", "ccd1", "ccd4", "lcyB", "lcyE", "psy", "pds", "zds",
    # precursor / mevalonate / MEP
    "HMG1", "HMG2", "tHMG1", "ERG9", "ERG20", "IDI1", "idi1", "ispA", "ISPDF", "dxs", "dxr", "ispD", "ispE", "ispF", "ispG", "ispH",
    "GGPS", "crtE2",
    # retinol / retinal
    "RDH12", "rdh12", "ALDH1A1", "aldh1a1", "ybbO",
    # common host engineering genes seen in these papers
    "MGA2", "ERG1", "MCM5", "DGK1",
    # strict title-case variants
    "CrtE", "CrtB", "CrtI", "CrtY", "CrtYB", "CrtZ", "CrtW", "Blh",
}

GENE_PATTERN_ALLOW = [
    re.compile(r"^(?:crt|car)[A-Za-z0-9]{1,4}$"),
    re.compile(r"^(?:blh|bkt|psy|pds|zds|dxs|dxr|idi1?|isp[a-h]?|ggps|ccd[0-9]*|lcy[BE]?|chyB|ybbO)$", re.I),
    re.compile(r"^(?:rdh|aldh)[A-Za-z0-9]{1,4}$", re.I),
    re.compile(r"^(?:HMG|ERG|MGA|MCM|DGK)[0-9A-Za-z]{1,4}$"),
    re.compile(r"^[a-z]{3,5}[A-Z][A-Za-z0-9]{0,3}$"),  # crtYB, chyB
    re.compile(r"^[A-Z][a-z]{2,4}[A-Z][A-Za-z0-9]{0,3}$"),  # CrtW
    re.compile(r"^[A-Z]{2,6}[0-9]{1,3}$"),  # HMG1, RDH12
]

# Tokens to suppress even if they look vaguely biological.
GENE_BLOCKLIST = {
    "carbon", "xylose", "acetate", "acetic", "acetyl", "acetyl-coa", "coa", "ggpp", "mva", "ipp", "dmapp",
    "retinol", "retinal", "carotene", "carotenoid", "aldehyde", "acid", "aroma", "designing", "high-yield",
    "crispr", "crispra", "scramble", "loxp", "de3", "bl21", "de", "novo", "phase", "two-phase",
    "food", "feed", "engineering", "biosynthesis", "production", "pathway", "metabolic", "synergistic",
    "culture", "strain", "host", "gene", "genes", "enzyme", "enzymes", "promoter", "yeast", "bacteria",
    "c5-c2", "c5", "c2", "carotene-", "cda-14", "g1637a", "yl19", "ysl4", "ygm97", "high", "yield",
}

PATHWAY_TAGS = {
    "carotenoid": {"crtE", "crtB", "crtI", "crtY", "crtYB", "crtZ", "crtW", "crtL", "crtR", "psy", "pds", "zds", "bkt", "BKT", "lcyB", "lcyE", "chyB", "Blh", "blh"},
    "mevalonate": {"HMG1", "HMG2", "tHMG1", "ERG9", "ERG20", "IDI1", "idi1", "MGA2", "ERG1"},
    "retinoid": {"RDH12", "rdh12", "ALDH1A1", "aldh1a1", "Blh", "blh", "ybbO"},
}

TITER_PATTERNS = [
    re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/L|g l-1|g l\^-1|mg/L|mg l-1|mg l\^-1|µg/L|ug/L)", re.I),
    re.compile(r"titer(?:s)?\s*(?:of|reached|was|were)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/L|g l-1|g l\^-1|mg/L|mg l-1|mg l\^-1|µg/L|ug/L)", re.I),
]

ITALIC_TAG_PATTERN = re.compile(r"</?i>", re.I)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
BINOMIAL_PATTERN = re.compile(r"\b([A-Z][a-z]{2,}\s+[a-z][a-z0-9-]{2,})\b")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = ITALIC_TAG_PATTERN.sub("", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    return text


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _contains_term(text: str, term: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", text.lower()) is not None


def extract_host(text: str, hosts_df: pd.DataFrame) -> tuple[str, float]:
    low = text.lower()
    for _, row in hosts_df.iterrows():
        aliases = [x.strip() for x in str(row.get("aliases", "")).split(";") if x.strip()]
        terms = [str(row.get("display_name", "")).strip(), *aliases]
        for term in terms:
            if term and _contains_term(low, term):
                confidence = 0.96 if term == str(row.get("display_name", "")).strip() else 0.88
                return str(row.get("display_name", "")).strip(), confidence
    return "", 0.0


def _normalize_gene_token(token: str) -> str:
    return normalize_spaces(token.strip(" ,;:.()[]{}"))


def _is_gene_candidate(token: str) -> bool:
    if not token:
        return False
    token = _normalize_gene_token(token)
    low = token.lower()
    if low in GENE_BLOCKLIST:
        return False
    if len(token) < 3 or len(token) > 10:
        return False
    if token in GENE_EXACT_ALLOW:
        return True
    return any(pat.fullmatch(token) for pat in GENE_PATTERN_ALLOW)


def _gene_contexts(text: str) -> list[str]:
    snippets: list[str] = []
    patterns = [
        re.compile(r"(?:gene|genes|enzyme|enzymes|homolog|overexpress(?:ed|ion)?|express(?:ed|ing)?|introduc(?:ed|ing)|harbor(?:ing)?|encoding)[:\s]+([^.;:]{0,160})", re.I),
        re.compile(r"(?:including|such as)\s+([^.;:]{0,120})", re.I),
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            snippets.append(m.group(1))
    title_sentence = text.split(".")[0]
    if title_sentence:
        snippets.append(title_sentence)
    snippets.append(text)
    return snippets


def extract_gene_candidates(text: str) -> tuple[list[str], float]:
    token_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9-]{1,9}\b")
    candidates: list[str] = []
    strong_hits = 0
    for snippet in _gene_contexts(text):
        for raw in token_pattern.findall(snippet):
            token = _normalize_gene_token(raw)
            if _is_gene_candidate(token):
                candidates.append(token)
                if token in GENE_EXACT_ALLOW or token.lower().startswith(("crt", "car", "blh", "rdh", "hmg", "erg", "idi", "isp", "dx")):
                    strong_hits += 1
    genes = _dedupe_keep_order(candidates)[:8]
    confidence = 0.0
    if genes:
        confidence = 0.95 if strong_hits >= 1 else 0.7
    return genes, confidence


def classify_pathway_type(genes: list[str]) -> str:
    gene_set = set(genes)
    scores = {k: len(gene_set & v) for k, v in PATHWAY_TAGS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def extract_titer(text: str) -> tuple[str, float]:
    for pat in TITER_PATTERNS:
        m = pat.search(text)
        if m:
            return f"{m.group('value')} {m.group('unit')}", 0.9
    return "", 0.0


def _canonical_organism_maps(safety_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, bool]]:
    aliases: dict[str, str] = {}
    allowed_map: dict[str, bool] = {}
    for _, row in safety_df.iterrows():
        canonical = str(row.get("organism_name", "")).strip()
        if not canonical:
            continue
        aliases[canonical.lower()] = canonical
        allowed_map[canonical.lower()] = bool(row.get("allowed"))
        for a in [x.strip() for x in str(row.get("aliases", "")).split(";") if x.strip()]:
            aliases[a.lower()] = canonical
    return aliases, allowed_map


def _curated_organism_hits(text: str, safety_df: pd.DataFrame) -> list[str]:
    aliases, _ = _canonical_organism_maps(safety_df)
    low = text.lower()
    hits: list[str] = []
    for alias, canonical in aliases.items():
        if _contains_term(low, alias):
            hits.append(canonical)
    return _dedupe_keep_order(hits)


def _gene_linked_donors(text: str, genes: list[str]) -> list[str]:
    hits: list[str] = []
    for gene in genes[:8]:
        patterns = [
            re.compile(rf"{re.escape(gene)}\s*(?:gene|enzyme|homolog)?\s+from\s+([A-Z][a-z]{{2,}}\s+[a-z][a-z0-9-]{{2,}})", re.I),
            re.compile(rf"from\s+([A-Z][a-z]{{2,}}\s+[a-z][a-z0-9-]{{2,}})\s+(?:{re.escape(gene)}|{re.escape(gene)}\s+gene)", re.I),
            re.compile(rf"([A-Z][a-z]{{2,}}\s+[a-z][a-z0-9-]{{2,}})[^.;:]{{0,35}}(?:{re.escape(gene)})", re.I),
        ]
        for pat in patterns:
            for m in pat.finditer(text):
                hits.append(normalize_spaces(m.group(1)))
    return _dedupe_keep_order(hits)


def extract_donor_organisms(text: str, safety_df: pd.DataFrame, genes: list[str], host: str) -> tuple[list[str], float]:
    aliases, _ = _canonical_organism_maps(safety_df)
    curated_hits = _curated_organism_hits(text, safety_df)
    gene_linked = _gene_linked_donors(text, genes)

    donors: list[str] = []
    for item in [*gene_linked, *curated_hits]:
        canonical = aliases.get(item.lower(), item)
        # Conservative: do not call the detected host a donor unless explicitly gene-linked and different from host is false.
        if host and canonical.lower() == host.lower():
            continue
        if BINOMIAL_PATTERN.fullmatch(canonical) or canonical in aliases.values():
            donors.append(canonical)

    donors = _dedupe_keep_order(donors)
    confidence = 0.0
    if gene_linked:
        confidence = 0.9
    elif donors:
        confidence = 0.65
    return donors[:6], confidence


def classify_donor_hits(donor_hits: list[str], safety_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    aliases, allowed_map = _canonical_organism_maps(safety_df)
    allowed: list[str] = []
    excluded: list[str] = []
    unknown: list[str] = []
    for hit in donor_hits:
        canonical = aliases.get(hit.lower(), hit)
        allowed_flag = allowed_map.get(canonical.lower())
        if allowed_flag is True:
            allowed.append(canonical)
        elif allowed_flag is False:
            excluded.append(canonical)
        else:
            unknown.append(canonical)
    return _dedupe_keep_order(allowed), _dedupe_keep_order(excluded), _dedupe_keep_order(unknown)


def _gate_field(items: list[str], confidence: float, threshold: float) -> list[str]:
    return items if confidence >= threshold else []


def extract_structured_fields(record: dict, hosts_df: pd.DataFrame, safety_df: pd.DataFrame) -> dict:
    title = clean_text(record.get("title", ""))
    abstract = clean_text(record.get("abstract", ""))
    text = normalize_spaces(f"{title}. {abstract}")

    host, host_conf = extract_host(title, hosts_df)
    if not host:
        host, host_conf = extract_host(abstract, hosts_df)

    genes, gene_conf = extract_gene_candidates(text)
    genes = _gate_field(genes, gene_conf, 0.6)

    donor_hits, donor_conf = extract_donor_organisms(text, safety_df, genes, host)
    donor_hits = _gate_field(donor_hits, donor_conf, 0.7)
    allowed_donors, excluded_donors, unknown_donors = classify_donor_hits(donor_hits, safety_df)
    titer, titer_conf = extract_titer(text)
    pathway_type = classify_pathway_type(genes)

    return {
        "extracted_host": host,
        "host_confidence": round(host_conf, 2),
        "extracted_genes": "; ".join(genes),
        "gene_confidence": round(gene_conf, 2),
        "likely_pathway_type": pathway_type,
        "extracted_donor_organisms": "; ".join(donor_hits),
        "donor_confidence": round(donor_conf, 2),
        "allowed_donor_hits": "; ".join(allowed_donors),
        "excluded_donor_hits": "; ".join(excluded_donors),
        "unknown_donor_hits": "; ".join(unknown_donors),
        "extracted_titer": titer,
        "titer_confidence": round(titer_conf, 2),
    }
