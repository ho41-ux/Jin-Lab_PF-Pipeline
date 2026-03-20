from pathlib import Path

import pandas as pd
import streamlit as st

from modules.extraction import extract_structured_fields
from modules.filtering import load_hosts, load_safety_table, score_record
from modules.gras import summarize_gras_hits
from modules.search import search_economic_evidence, search_multi_source

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="SafePath PF Search v9.0.2", layout="wide")

hosts_df = load_hosts(DATA_DIR / "hosts.csv")
safety_df = load_safety_table(DATA_DIR / "organism_safety.csv")
allowed_hosts = hosts_df.loc[hosts_df["allowed"].astype(bool), "display_name"].tolist()

TEA_KEYWORDS = [
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
]

LCA_KEYWORDS = [
    "life cycle assessment",
    "life-cycle assessment",
    "lca",
    "environmental impact",
    "environmental assessment",
    "carbon footprint",
    "sustainability analysis",
    "sustainability",
    "cradle-to-gate",
    "cradle to gate",
]

PRICE_SIGNAL_MAP = {
    "single-cell protein": "~$",
    "single cell protein": "~$",
    "mycoprotein": "~$",
    "ethanol": "~$",
    "lactic acid": "~$",
    "fatty alcohol": "~$$",
    "fatty alcohols": "~$$",
    "farnesene": "~$$",
    "organic acid": "~$$",
    "vanillin": "~$$$",
    "beta-carotene": "~$$$",
    "beta carotene": "~$$$",
    "lycopene": "~$$$",
    "resveratrol": "~$$$",
    "ovalbumin": "~$$$$",
    "casein": "~$$$$",
    "lactoferrin": "~$$$$",
}


def detect_keywords(text: str, keywords: list[str]) -> bool:
    text = (text or "").lower()
    return any(k in text for k in keywords)


def infer_production_type(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["secreted", "extracellular", "culture supernatant", "supernatant"]):
        return "secreted"
    if any(k in t for k in ["biomass", "single-cell protein", "single cell protein", "mycoprotein", "whole-cell biomass", "cell biomass"]):
        return "biomass-associated"
    if any(k in t for k in ["intracellular", "cell lysate", "cell extract", "intracellularly", "cell pellet"]):
        return "intracellular"
    return "unclear"


def get_market_signal(molecule: str, molecule_class: str) -> str:
    m = (molecule or "").strip().lower()
    if m in PRICE_SIGNAL_MAP:
        return PRICE_SIGNAL_MAP[m]
    if molecule_class == "protein":
        return "~$$$$"
    if molecule_class == "biomass/SCP":
        return "~$"
    if molecule_class == "metabolite":
        return "~$$$"
    return "—"


def classify_curated_host_status(host_value: object, allowed: list[str]) -> str:
    host = str(host_value).strip() if pd.notna(host_value) else ""
    if not host:
        return "Unknown"
    if host in allowed:
        return "Allowed host"
    return "Not on allowed host list"


def ensure_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "title": "",
        "abstract": "",
        "authors": "",
        "journal": "",
        "year": "",
        "doi": "",
        "pmid": "",
        "url": "",
        "retrieved_from": "",
        "source_query": "",
        "article_types": "",
        "is_review": False,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def add_display_columns(df: pd.DataFrame, molecule: str, resolved_class: str) -> pd.DataFrame:
    df = ensure_basic_columns(df.copy())
    production_text = df["title"].fillna("") + " " + df["abstract"].fillna("")
    df["Host organism"] = df.get("extracted_host", "").fillna("").replace("", "Unknown")
    df["Curated host status"] = df["Host organism"].apply(lambda x: classify_curated_host_status(x, allowed_hosts))
    df["Titer"] = df.get("extracted_titer", "").fillna("").replace("", "—")

    def normalize_titer_label(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        s = str(value).strip()
        if not s or s == "nan":
            return "—"
        for old, new in [("equiv", ""), ("Equivalent", ""), ("equivalent", ""), ("g l-1", "g/L"), ("g l^-1", "g/L"), ("mg l-1", "mg/L"), ("mg l^-1", "mg/L"), ("µg l-1", "µg/L"), ("ug l-1", "ug/L"), ("ug l^-1", "ug/L")]:
            s = s.replace(old, new)
        s = " ".join(s.split())
        return s if s else "—"

    df["Titer"] = df["Titer"].apply(normalize_titer_label)
    df["Production type"] = production_text.apply(infer_production_type)
    df["TEA mentioned"] = production_text.apply(lambda t: "Yes" if detect_keywords(t, TEA_KEYWORDS) else "No")
    df["LCA mentioned"] = production_text.apply(lambda t: "Yes" if detect_keywords(t, LCA_KEYWORDS) else "No")
    df["Pathway genes involved"] = df.get("extracted_genes", "").fillna("").replace("", "—")
    df["Paper title"] = df["title"].fillna("")
    df["Year"] = df["year"]
    df["Authors"] = df["authors"].fillna("")
    df["DOI"] = df["doi"].fillna("").replace("", "—")
    df["URL"] = df["url"].fillna("").replace("", "—")
    return df


st.title("Jin Lab Precision Fermentation Search Tool v9.0.2")

with st.expander("What v9.0.2 changes"):
    st.markdown(
        """
        - Uses **two retrieval philosophies**: **Focused** and **Exploratory**.
        - Keeps **FDA GRAS Notice Inventory evidence** in a separate table.
        - Keeps **Production type** in the main table and moves **Market signal** to the summary layer.
        - Keeps a broader secondary **economic evidence search** for TEA / LCA using more permissive techno-economic and sustainability queries.
        - Adds a cleaner summary with **Top host** and **Max titer observed**.
        """
    )

with st.sidebar:
    st.header("Retrieval")
    molecule_class = st.selectbox(
        "Molecule class",
        options=["auto", "metabolite", "protein", "biomass/SCP", "unknown"],
        index=0,
    )
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=[
            ("exploratory", "Exploratory (broad, permissive)"),
            ("focused", "Focused (clean, strict)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]
    preferred_host = st.selectbox(
        "Preferred host",
        options=["All allowed hosts", *allowed_hosts],
        index=0,
    )
    broad_mode = st.checkbox("Broad search mode", value=True)
    exclude_reviews = st.checkbox("Exclude reviews", value=True)
    include_europe_pmc = st.checkbox("Search Europe PMC", value=True)
    include_pubmed = st.checkbox("Search PubMed", value=True)
    page_size_per_query = st.slider("Papers per query per source", 5, 100, 50, 5)

    st.markdown("---")
    st.header("Constraints")
    enforce_allowed_hosts = st.checkbox("Allowed hosts only", value=True)
    exclude_disallowed_donors = st.checkbox("Exclude excluded donor organisms", value=True)
    flag_unknown_donors = st.checkbox("Flag unknown donor organisms", value=False)

    st.markdown("---")
    st.subheader("Curated allowed hosts")
    st.dataframe(hosts_df[["display_name", "allowed"]], width="stretch", hide_index=True)

with st.form("search_form"):
    molecule = st.text_input("Target molecule", placeholder="e.g., beta-carotene, ovalbumin, vanillin")
    extra_terms = st.text_input(
        "Optional extra terms",
        placeholder="e.g., Yarrowia lipolytica, secretion, productivity, substrate inhibition",
    )
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    if not molecule.strip():
        st.error("Enter a target molecule first.")
        st.stop()

    with st.spinner("Searching, deduplicating, and extracting..."):
        records, queries_used, resolved_class = search_multi_source(
            molecule=molecule.strip(),
            molecule_class=molecule_class,
            retrieval_mode=retrieval_mode,
            preferred_host=preferred_host,
            extra_terms=extra_terms.strip(),
            broad_mode=broad_mode,
            include_europe_pmc=include_europe_pmc,
            include_pubmed=include_pubmed,
            page_size_per_query=page_size_per_query,
            exclude_reviews=exclude_reviews,
        )
        econ_records, econ_queries_used, _ = search_economic_evidence(
            molecule=molecule.strip(),
            molecule_class=molecule_class,
            preferred_host=preferred_host,
            extra_terms=extra_terms.strip(),
            broad_mode=True,
            include_europe_pmc=include_europe_pmc,
            include_pubmed=include_pubmed,
            page_size_per_query=max(10, min(20, page_size_per_query)),
            exclude_reviews=False,
        )

    if not records:
        st.warning("No papers found.")
        st.stop()

    enriched = []
    for rec in records:
        extracted = extract_structured_fields(rec, hosts_df, safety_df)
        rec.update(extracted)
        rec["allowed_host_match"] = bool(rec.get("extracted_host"))
        rec["has_excluded_donor"] = bool(rec.get("excluded_donor_hits"))
        rec["has_unknown_donor"] = bool(rec.get("unknown_donor_hits"))
        rec["pass_constraints"] = (
            (not enforce_allowed_hosts or rec["allowed_host_match"])
            and (not exclude_disallowed_donors or not rec["has_excluded_donor"])
            and (not flag_unknown_donors or not rec["has_unknown_donor"])
        )
        score_text = f"{rec.get('title', '')} {rec.get('abstract', '')} {rec.get('extracted_genes', '')}"
        rec["score"] = score_record(
            allowed_host_match=rec["allowed_host_match"],
            excluded_count=1 if rec["has_excluded_donor"] else 0,
            text=score_text,
        )
        if preferred_host != "All allowed hosts" and rec.get("extracted_host") == preferred_host:
            rec["score"] += 2
        if rec.get("extracted_titer"):
            rec["score"] += 1
        enriched.append(rec)

    df = add_display_columns(pd.DataFrame(enriched), molecule.strip(), resolved_class)

    econ_enriched = []
    for rec in econ_records:
        rec_copy = rec.copy()
        extracted = extract_structured_fields(rec_copy, hosts_df, safety_df)
        rec_copy.update(extracted)
        econ_enriched.append(rec_copy)
    econ_df = add_display_columns(pd.DataFrame(econ_enriched), molecule.strip(), resolved_class)
    econ_df = econ_df.drop_duplicates(subset=[c for c in ["doi", "pmid", "title"] if c in econ_df.columns])

    unique_hosts = sorted({h for h in df["Host organism"].dropna().astype(str).tolist() if h and h != "Unknown"})
    gras_lookup = {}
    gras_rows = []
    for host in unique_hosts:
        host_summary = summarize_gras_hits(host)
        combo_summary = summarize_gras_hits(f"{host} {molecule.strip()}")
        gras_lookup[host] = {
            "GRAS (host)": "Yes" if host_summary["has_hit"] else "No hit found",
            "GRAS (host + target)": "Yes" if combo_summary["has_hit"] else "No hit found",
        }
        for hit in host_summary["hits"]:
            gras_rows.append({
                "Host organism": host,
                "Target molecule": molecule.strip(),
                "Evidence type": "Host",
                "GRN": hit.get("grn", ""),
                "Substance": hit.get("substance", ""),
                "Inventory date": hit.get("inventory_date", ""),
                "FDA response": hit.get("response", ""),
                "Notice URL": hit.get("notice_url", ""),
            })
        for hit in combo_summary["hits"]:
            gras_rows.append({
                "Host organism": host,
                "Target molecule": molecule.strip(),
                "Evidence type": "Host + target",
                "GRN": hit.get("grn", ""),
                "Substance": hit.get("substance", ""),
                "Inventory date": hit.get("inventory_date", ""),
                "FDA response": hit.get("response", ""),
                "Notice URL": hit.get("notice_url", ""),
            })

    df["GRAS (host)"] = df["Host organism"].map(lambda h: gras_lookup.get(h, {}).get("GRAS (host)", "Unknown" if h == "Unknown" else "No hit found"))
    df["GRAS (host + target)"] = df["Host organism"].map(lambda h: gras_lookup.get(h, {}).get("GRAS (host + target)", "Unknown" if h == "Unknown" else "No hit found"))

    df = df.sort_values(by=["Host organism", "pass_constraints", "year", "score"], ascending=[True, False, False, False])

    market_signal = get_market_signal(molecule.strip(), resolved_class)
    top_host_counts = df[~df["Host organism"].eq("Unknown")]["Host organism"].value_counts()
    if not top_host_counts.empty:
        top_host_label = f"{top_host_counts.index[0]} ({int(top_host_counts.iloc[0])})"
    else:
        top_host_label = "—"

    def titer_sort_value(value: object) -> float | None:
        if pd.isna(value):
            return None
        s = str(value)
        import re
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
        if not m:
            return None
        val = float(m.group(1))
        low = s.lower()
        if "mg/l" in low:
            return val / 1000.0
        if "g/l" in low:
            return val
        return val

    df["_titer_sort"] = df["Titer"].apply(titer_sort_value)

    st.subheader("📊 Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Papers found", len(df))
    s2.metric("Passing current filters", int(df["pass_constraints"].sum()))
    s3.metric("Molecule class", resolved_class)
    s4.metric("Market signal", market_signal)

    p1 = st.columns(1)[0]
    p1.metric("Top host (most papers)", top_host_label)

    tea_df = econ_df[econ_df["TEA mentioned"] == "Yes"].copy()
    lca_df = econ_df[econ_df["LCA mentioned"] == "Yes"].copy()

    st.subheader("📊 Economic Evidence Summary")
    e1, e2 = st.columns(2)
    e1.metric("TEA papers", f"{len(tea_df)} / {len(econ_df)}")
    e2.metric("LCA papers", f"{len(lca_df)} / {len(econ_df)}")
    st.caption("These counts come from a broader secondary economic-evidence search with more permissive techno-economic and sustainability queries, not only from the main host-first results table.")

    with st.expander("Queries used", expanded=False):
        st.write("**Main search**")
        for q in queries_used:
            st.code(q)

        st.write("**Economic evidence search**")
        for q in econ_queries_used:
            st.code(q)

    host_summary = (
        df.groupby("Host organism", dropna=False)
        .agg(Papers=("Paper title", "count"))
        .reset_index()
    )
    if not host_summary.empty:
        host_summary = host_summary.sort_values(by=["Papers", "Host organism"], ascending=[False, True]).reset_index(drop=True)
        unknown_mask = host_summary["Host organism"].eq("Unknown")
        if unknown_mask.any():
            host_summary = pd.concat([host_summary[~unknown_mask], host_summary[unknown_mask]], ignore_index=True)

    st.subheader("Results by host organism")
    st.dataframe(host_summary, width="stretch", hide_index=True)

    display_cols = [
        "Host organism",
        "Titer",
        "Production type",
        "Curated host status",
        "GRAS (host)",
        "GRAS (host + target)",
        "TEA mentioned",
        "LCA mentioned",
        "Pathway genes involved",
        "Paper title",
        "Year",
        "Authors",
        "DOI",
        "URL",
    ]

    def safe_display_cols(frame: pd.DataFrame, cols: list[str]) -> list[str]:
        return [c for c in cols if c in frame.columns]

    st.subheader("Grouped by host organism")
    for host_name in host_summary["Host organism"].tolist():
        subset = df[df["Host organism"] == host_name].copy()
        with st.expander(f"{host_name} ({len(subset)} papers)", expanded=(host_name != "Unknown" and len(subset) <= 8)):
            st.dataframe(subset[safe_display_cols(subset, display_cols)], width="stretch", hide_index=True)

    st.subheader("🔬 TEA-related papers")
    if not tea_df.empty:
        st.dataframe(tea_df[safe_display_cols(tea_df, display_cols)], width="stretch", hide_index=True)
    else:
        st.info("No TEA-related papers were detected from the broader economic-evidence search.")

    st.subheader("🌱 LCA-related papers")
    if not lca_df.empty:
        st.dataframe(lca_df[safe_display_cols(lca_df, display_cols)], width="stretch", hide_index=True)
    else:
        st.info("No LCA-related papers were detected from the broader economic-evidence search.")

    st.subheader("Structured results")
    st.dataframe(df[safe_display_cols(df, display_cols)], width="stretch", hide_index=True)

    gras_df = pd.DataFrame(gras_rows)
    st.subheader("GRAS Notice Evidence")
    if not gras_df.empty:
        gras_df = gras_df.sort_values(by=["Host organism", "Evidence type", "GRN"], ascending=[True, True, False]).drop_duplicates()
        st.dataframe(gras_df, width="stretch", hide_index=True)
    else:
        st.info("No GRAS notice hits were found for the detected hosts or for host + target searches.")

    csv_cols = [
        c
        for c in display_cols
        + [
            "pass_constraints",
            "score",
            "retrieved_from",
            "source_query",
            "article_types",
            "is_review",
            "abstract",
            "journal",
            "pmid",
            "extracted_host",
            "extracted_genes",
            "extracted_titer",
            "Curated host status",
            "GRAS (host)",
            "GRAS (host + target)",
        ]
        if c in df.columns
    ]
    csv_bytes = df[csv_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download structured results as CSV",
        data=csv_bytes,
        file_name=f"safepath_{molecule.strip().replace(' ', '_')}_v9.csv",
        mime="text/csv",
    )

    passing_df = df[df["pass_constraints"]].copy()
    st.subheader("Passing papers")
    if not passing_df.empty:
        for _, row in passing_df.head(12).iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['title']}**")
                meta = " | ".join(
                    [
                        x
                        for x in [
                            str(row.get("year", "")),
                            row.get("journal", ""),
                            row.get("doi", ""),
                            row.get("retrieved_from", ""),
                            row.get("article_types", ""),
                        ]
                        if x
                    ]
                )
                if meta:
                    st.caption(meta)
                st.write(f"**Host organism:** {row.get('extracted_host') or '—'} (confidence {row.get('host_confidence', 0):.2f})")
                st.write(f"**Extracted genes:** {row.get('extracted_genes') or '—'} (confidence {row.get('gene_confidence', 0):.2f})")
                st.write(f"**Likely pathway type:** {row.get('likely_pathway_type') or 'unknown'}")
                st.write(f"**Extracted donor organisms:** {row.get('extracted_donor_organisms') or '—'} (confidence {row.get('donor_confidence', 0):.2f})")
                st.write(f"**Extracted titer:** {row.get('extracted_titer') or '—'} (confidence {row.get('titer_confidence', 0):.2f})")
                st.write(f"**Review flag:** {'Yes' if row.get('is_review') else 'No'}")
                if row.get("excluded_donor_hits"):
                    st.write(f"**Excluded donor hits:** {row['excluded_donor_hits']}")
                if row.get("unknown_donor_hits"):
                    st.write(f"**Unknown donor hits:** {row['unknown_donor_hits']}")
                st.write(row.get("abstract") or "No abstract available.")
                if row.get("url"):
                    st.markdown(f"[Open paper record]({row['url']})")
    else:
        st.info("No papers passed the current constraints. Try a broader retrieval mode, more optional terms, or relax one filter.")

    rejected_df = df[~df["pass_constraints"]].copy()
    if not rejected_df.empty:
        st.subheader("Rejected by current filters")
        reject_cols = [c for c in ["title", "retrieved_from", "extracted_host", "excluded_donor_hits", "unknown_donor_hits", "url"] if c in rejected_df.columns]
        st.dataframe(rejected_df[reject_cols], width="stretch", hide_index=True)
