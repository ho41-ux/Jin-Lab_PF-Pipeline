from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Iterable

import requests

EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

CLASS_TERMS = {
    "metabolite": [
        "metabolic engineering",
        "heterologous",
        "biosynthesis",
        "production",
        "engineered",
        "synthetic biology",
        "precision fermentation",
    ],
    "protein": ["recombinant", "expression", "secreted", "secretion", "engineered", "production"],
    "biomass/SCP": ["fermentation", "biomass", "single-cell protein", "mycoprotein", "cultivation", "production"],
    "unknown": ["biosynthesis", "production", "engineered"],
}

PROTEIN_HINTS = {"ovalbumin", "casein", "lactoferrin", "albumin", "leghemoglobin", "hemoglobin", "lysozyme"}
BIOMASS_HINTS = {"single-cell protein", "mycoprotein", "yeast extract", "biomass"}

FAMILY_TERMS = {
    "beta-carotene": ["carotenoid", "carotenoids", "lycopene", "phytoene", "phaffia rhodozyma", "xanthophyllomyces dendrorhous"],
    "beta carotene": ["carotenoid", "carotenoids", "lycopene", "phytoene", "phaffia rhodozyma", "xanthophyllomyces dendrorhous"],
    "lycopene": ["carotenoid", "carotenoids", "beta-carotene", "phytoene", "phaffia rhodozyma", "xanthophyllomyces dendrorhous"],
    "vanillin": ["vanillic acid", "ferulic acid", "aroma compound"],
    "farnesene": ["sesquiterpene", "terpenoid", "mevalonate"],
    "fatty alcohol": ["oleochemical", "fatty acid", "acyl-CoA"],
    "fatty alcohols": ["oleochemical", "fatty acid", "acyl-CoA"],
    "ovalbumin": ["egg white protein", "recombinant protein"],
}

ECONOMIC_KEYWORD_GROUPS = {
    "TEA": [
        "techno-economic",
        "techno economic",
        "economic analysis",
        "economic feasibility",
        "cost analysis",
        "cost estimation",
        "process economics",
        "minimum selling price",
        "techno-economic constraints",
        "economic assessment",
    ],
    "LCA": [
        "life cycle assessment",
        "life-cycle assessment",
        "environmental assessment",
        "environmental impact",
        "carbon footprint",
        "sustainability analysis",
        "sustainability",
        "cradle-to-gate",
        "cradle to gate",
    ],
}


def infer_molecule_class(molecule: str, chosen_class: str) -> str:
    if chosen_class != "auto":
        return chosen_class
    m = molecule.lower().strip()
    if any(x in m for x in BIOMASS_HINTS):
        return "biomass/SCP"
    if any(x in m for x in PROTEIN_HINTS):
        return "protein"
    return "metabolite"


def _clean_query_term(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def _or_group(terms: Iterable[str]) -> str:
    terms = [_clean_query_term(t) for t in terms if _clean_query_term(t)]
    if not terms:
        return ""
    if len(terms) == 1:
        return terms[0]
    return "(" + " OR ".join(terms) + ")"


def build_queries(
    molecule: str,
    molecule_class: str,
    retrieval_mode: str,
    preferred_host: str,
    extra_terms: str = "",
    broad_mode: bool = True,
) -> list[str]:
    molecule = _clean_query_term(molecule)
    extra_terms = _clean_query_term(extra_terms)
    class_terms = CLASS_TERMS.get(molecule_class, CLASS_TERMS["unknown"])

    preferred_host = _clean_query_term(preferred_host)
    host_terms: list[str] = []
    if preferred_host and preferred_host.lower() != "all allowed hosts":
        host_terms.extend([preferred_host])
        if " " in preferred_host:
            host_terms.append(preferred_host.split()[0])
    else:
        if molecule_class == "protein":
            host_terms.extend(["yeast", "microbial", "fermentation"])
        elif molecule_class == "metabolite":
            host_terms.extend(["yeast", "microbial", "fermentation"])
    host_part = _or_group(host_terms)

    queries: list[str] = []
    if retrieval_mode == "focused":
        pieces = [molecule]
        if host_part:
            pieces.append(host_part)
        pieces.append(_or_group(class_terms))
        queries.append(" AND ".join([p for p in pieces if p]))
    else:  # exploratory
        rel = FAMILY_TERMS.get(molecule.lower(), []) if broad_mode else []
        target_group = _or_group([molecule, *rel] if rel else [molecule])
        exploratory_context = class_terms + [
            "synthetic biology",
            "precision fermentation",
            "sustainability",
            "environmental assessment",
            "techno-economic",
            "life cycle assessment",
        ]
        pieces = [target_group]
        if host_part:
            pieces.append(host_part)
        pieces.append(_or_group(exploratory_context))
        queries.append(" AND ".join([p for p in pieces if p]))

        # Host-optional broad biological query to improve recall on non-obvious papers.
        broad_pieces = [target_group, _or_group(exploratory_context)]
        queries.append(" AND ".join([p for p in broad_pieces if p]))

    if broad_mode:
        fallback_pieces = [molecule, _or_group(class_terms)]
        if host_part and retrieval_mode == "focused":
            fallback_pieces.insert(1, host_part)
        queries.append(" AND ".join([p for p in fallback_pieces if p]))

    if extra_terms:
        augmented: list[str] = []
        for q in queries or [molecule]:
            augmented.append(f"({q}) AND ({extra_terms})")
        queries.extend(augmented)

    deduped: list[str] = []
    seen = set()
    for q in queries:
        q = _clean_query_term(q)
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped[:8]


def build_economic_queries(
    molecule: str,
    molecule_class: str,
    preferred_host: str = "All allowed hosts",
    extra_terms: str = "",
    broad_mode: bool = True,
) -> list[str]:
    molecule = _clean_query_term(molecule)
    extra_terms = _clean_query_term(extra_terms)
    preferred_host = _clean_query_term(preferred_host)

    family_terms = FAMILY_TERMS.get(molecule.lower(), []) if broad_mode else []
    target_group = _or_group([molecule, *family_terms] if family_terms else [molecule])

    host_terms: list[str] = []
    if preferred_host and preferred_host.lower() != "all allowed hosts":
        host_terms.extend([preferred_host])
        if " " in preferred_host:
            host_terms.append(preferred_host.split()[0])
    host_group = _or_group(host_terms)

    queries: list[str] = []
    for terms in ECONOMIC_KEYWORD_GROUPS.values():
        econ_group = _or_group(terms)
        queries.append(f"{target_group} AND {econ_group}")
        if host_group:
            queries.append(f"{target_group} AND {host_group} AND {econ_group}")

    if extra_terms:
        queries.extend([f"({q}) AND ({extra_terms})" for q in list(queries)])

    deduped: list[str] = []
    seen = set()
    for q in queries:
        q = _clean_query_term(q)
        if q and q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped[:12]


def _article_type_to_review(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in ["review", "mini-review", "perspective", "opinion", "survey"])


def search_europe_pmc(query: str, page_size: int = 25) -> list[dict]:
    params = {"query": query, "format": "json", "pageSize": page_size, "resultType": "core"}
    resp = requests.get(EUROPE_PMC_URL, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("resultList", {}).get("result", [])
    records: list[dict] = []
    for item in results:
        article_types = "; ".join(item.get("pubTypeList", {}).get("pubType", []) or [])
        pmid = item.get("pmid", "")
        doi = item.get("doi", "")
        url = f"https://europepmc.org/article/MED/{pmid}" if pmid else ""
        records.append(
            {
                "title": html.unescape(item.get("title", "") or ""),
                "abstract": html.unescape(item.get("abstractText", "") or ""),
                "authors": item.get("authorString", "") or "",
                "journal": item.get("journalTitle", "") or "",
                "year": item.get("pubYear", "") or "",
                "doi": doi,
                "pmid": pmid,
                "url": url,
                "retrieved_from": "Europe PMC",
                "source_query": query,
                "article_types": article_types,
                "is_review": _article_type_to_review(article_types + " " + (item.get("title", "") or "")),
            }
        )
    return records


def _pubmed_search_pmids(query: str, retmax: int = 25) -> list[str]:
    params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json", "sort": "relevance"}
    resp = requests.get(PUBMED_ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", []) or []


def _pubmed_fetch_records(pmids: list[str], query: str) -> list[dict]:
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    resp = requests.get(PUBMED_EFETCH_URL, params=params, timeout=45)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    records: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_el = medline.find("Article") if medline is not None else None
        if article_el is None:
            continue
        title = "".join(article_el.findtext("ArticleTitle", default="") or "")
        abstract_parts = ["".join(abst.itertext()) for abst in article_el.findall("Abstract/AbstractText")]
        abstract = " ".join([p.strip() for p in abstract_parts if p.strip()])
        journal = article_el.findtext("Journal/Title", default="") or ""
        year = article_el.findtext("Journal/JournalIssue/PubDate/Year", default="") or ""
        authors = []
        for author in article_el.findall("AuthorList/Author"):
            lastname = author.findtext("LastName", default="")
            initials = author.findtext("Initials", default="")
            coll = author.findtext("CollectiveName", default="")
            if coll:
                authors.append(coll)
            elif lastname:
                authors.append((lastname + (f" {initials}" if initials else "")).strip())
        pmid = medline.findtext("PMID", default="") if medline is not None else ""
        doi = ""
        for aid in article_el.findall("ELocationID"):
            if aid.attrib.get("EIdType") == "doi":
                doi = "".join(aid.itertext()).strip()
                break
        pub_types = [pt.text.strip() for pt in article_el.findall("PublicationTypeList/PublicationType") if pt.text]
        article_types = "; ".join(pub_types)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        records.append(
            {
                "title": html.unescape(title),
                "abstract": html.unescape(abstract),
                "authors": ", ".join(authors),
                "journal": journal,
                "year": year,
                "doi": doi,
                "pmid": pmid,
                "url": url,
                "retrieved_from": "PubMed",
                "source_query": query,
                "article_types": article_types,
                "is_review": _article_type_to_review(article_types + " " + title),
            }
        )
    return records


def search_pubmed(query: str, page_size: int = 25) -> list[dict]:
    pmids = _pubmed_search_pmids(query, retmax=page_size)
    return _pubmed_fetch_records(pmids, query)


def deduplicate_records(records: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for rec in records:
        key = (rec.get("doi") or "").strip().lower()
        if not key:
            key = (rec.get("pmid") or "").strip()
        if not key:
            key = re.sub(r"\W+", " ", (rec.get("title") or "").lower()).strip()
        if key in merged:
            existing = merged[key]
            sources = sorted(set(filter(None, (existing.get("retrieved_from", "") + "; " + rec.get("retrieved_from", "")).split("; "))))
            existing["retrieved_from"] = "; ".join(sources)
            sq = sorted(set(filter(None, (existing.get("source_query", "") + "; " + rec.get("source_query", "")).split("; "))))
            existing["source_query"] = "; ".join(sq)
            if not existing.get("abstract") and rec.get("abstract"):
                existing["abstract"] = rec["abstract"]
            if not existing.get("doi") and rec.get("doi"):
                existing["doi"] = rec["doi"]
            if not existing.get("pmid") and rec.get("pmid"):
                existing["pmid"] = rec["pmid"]
            existing["is_review"] = bool(existing.get("is_review")) or bool(rec.get("is_review"))
            existing_types = sorted(set(filter(None, (existing.get("article_types", "") + "; " + rec.get("article_types", "")).split("; "))))
            existing["article_types"] = "; ".join(existing_types)
        else:
            merged[key] = rec.copy()
    return list(merged.values())


def search_multi_source(
    molecule: str,
    molecule_class: str = "auto",
    retrieval_mode: str = "target_host_class",
    preferred_host: str = "All allowed hosts",
    extra_terms: str = "",
    broad_mode: bool = True,
    include_europe_pmc: bool = True,
    include_pubmed: bool = True,
    page_size_per_query: int = 25,
    exclude_reviews: bool = True,
) -> tuple[list[dict], list[str], str]:
    resolved_class = infer_molecule_class(molecule, molecule_class)
    queries = build_queries(
        molecule=molecule,
        molecule_class=resolved_class,
        retrieval_mode=retrieval_mode,
        preferred_host=preferred_host,
        extra_terms=extra_terms,
        broad_mode=broad_mode,
    )
    all_records: list[dict] = []
    for q in queries:
        if include_europe_pmc:
            try:
                all_records.extend(search_europe_pmc(q, page_size=page_size_per_query))
            except Exception:
                pass
        if include_pubmed:
            try:
                all_records.extend(search_pubmed(q, page_size=page_size_per_query))
            except Exception:
                pass
    records = deduplicate_records(all_records)
    if exclude_reviews:
        records = [r for r in records if not r.get("is_review")]
    return records, queries, resolved_class


def search_economic_evidence(
    molecule: str,
    molecule_class: str = "auto",
    preferred_host: str = "All allowed hosts",
    extra_terms: str = "",
    broad_mode: bool = True,
    include_europe_pmc: bool = True,
    include_pubmed: bool = True,
    page_size_per_query: int = 20,
    exclude_reviews: bool = False,
) -> tuple[list[dict], list[str], str]:
    resolved_class = infer_molecule_class(molecule, molecule_class)
    queries = build_economic_queries(
        molecule=molecule,
        molecule_class=resolved_class,
        preferred_host=preferred_host,
        extra_terms=extra_terms,
        broad_mode=broad_mode,
    )
    all_records: list[dict] = []
    for q in queries:
        if include_europe_pmc:
            try:
                all_records.extend(search_europe_pmc(q, page_size=page_size_per_query))
            except Exception:
                pass
        if include_pubmed:
            try:
                all_records.extend(search_pubmed(q, page_size=page_size_per_query))
            except Exception:
                pass
    records = deduplicate_records(all_records)
    if exclude_reviews:
        records = [r for r in records if not r.get("is_review")]
    return records, queries, resolved_class
