"""Streamlit frontend for Financial Document Q&A."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import requests
import streamlit as st

API_BASE = os.environ.get("RAG_API_BASE", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120

DOC_TYPES = ["10-K", "10-Q", "earnings_transcript"]


def _api_url(path: str) -> str:
    return f"{API_BASE}{path}"


def _fetch_documents() -> list[dict[str, Any]]:
    try:
        response = requests.get(_api_url("/documents"), timeout=30)
    except requests.RequestException as exc:
        st.error(
            f"Cannot reach the API at `{API_BASE}`. Start the backend with "
            f"`uvicorn api.main:app --reload` and try again.\n\nDetails: {exc}"
        )
        st.stop()
    if response.status_code != 200:
        st.error(f"GET /documents failed: HTTP {response.status_code}\n\n{response.text}")
        st.stop()
    data = response.json()
    if not isinstance(data, list):
        st.error("Unexpected /documents response shape.")
        st.stop()
    return data


def _singleton_filter(label: str, values: list[Any]) -> Any | None:
    """API accepts one value per filter; mirror multiselect UX with a clear caption."""
    if not values:
        return None
    if len(values) > 1:
        st.sidebar.caption(
            f"⚠ *{label}:* multiple selected — sending **{values[0]}** only (API is single-value)."
        )
        return values[0]
    return values[0]


def _normalize_rerank_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _maybe_warn_empty(answer: str, chunk_count: int) -> None:
    low = answer.lower()
    if chunk_count == 0:
        st.warning("No chunks were retrieved for this query.")
        return
    heuristics = (
        "no chunks" in low,
        "not contained" in low,
        "not in the sources" in low,
        "insufficient" in low,
    )
    if any(heuristics):
        st.warning("The answer may be ungrounded or the model reported missing sources.")


def main() -> None:
    st.set_page_config(
        page_title="Financial Document Q&A",
        layout="wide",
    )
    st.title("Financial Document Q&A")
    st.caption("RAG over SEC filings with metadata filtering")

    if "documents" not in st.session_state:
        st.session_state.documents = _fetch_documents()

    docs: list[dict[str, Any]] = st.session_state.documents

    companies = sorted(
        {str(d["company"]) for d in docs if d.get("company")},
        key=lambda x: x.lower(),
    )
    sectors = sorted(
        {str(d["sector"]) for d in docs if d.get("sector")},
        key=lambda x: x.lower(),
    )
    years = sorted(
        {int(d["fiscal_year"]) for d in docs if d.get("fiscal_year") is not None},
        reverse=True,
    )

    with st.sidebar:
        st.header("Filters")
        st.caption("All optional. Each field maps to one API filter value.")

        sel_companies = st.multiselect("Company", options=companies, default=[])
        sel_doc_types = st.multiselect("Doc Type", options=DOC_TYPES, default=[])
        sel_sectors = st.multiselect("Sector", options=sectors, default=[])
        sel_years = st.multiselect("Fiscal Year", options=years, default=[], format_func=str)

        st.subheader("Filing date range")
        c1, c2 = st.columns(2)
        with c1:
            d_from_raw = st.date_input(
                "From",
                value=None,
                key="filing_from",
                help="Leave empty for no start bound",
            )
        with c2:
            d_to_raw = st.date_input(
                "To",
                value=None,
                key="filing_to",
                help="Leave empty for no end bound",
            )

    filters: dict[str, Any] = {}
    company_val = _singleton_filter("Company", sel_companies)
    if company_val is not None:
        filters["company"] = company_val
    doc_type_val = _singleton_filter("Doc type", sel_doc_types)
    if doc_type_val is not None:
        filters["doc_type"] = doc_type_val
    sector_val = _singleton_filter("Sector", sel_sectors)
    if sector_val is not None:
        filters["sector"] = sector_val
    year_val = _singleton_filter("Fiscal year", sel_years)
    if year_val is not None:
        filters["fiscal_year"] = int(year_val)

    if isinstance(d_from_raw, date):
        filters["filing_date_from"] = d_from_raw.isoformat()
    if isinstance(d_to_raw, date):
        filters["filing_date_to"] = d_to_raw.isoformat()

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    st.text_area(
        "Ask a question about the filings",
        height=120,
        key="query_area",
        placeholder="e.g. How did gross margin trend year over year?",
    )

    if st.button("Search", type="primary"):
        stripped = str(st.session_state.get("query_area", "")).strip()
        if not stripped:
            st.error("Enter a question before searching.")
        else:
            payload: dict[str, Any] = {
                "query": stripped,
                "vector_top_k": 20,
                "rerank_top_k": 5,
            }
            if filters:
                payload["filters"] = filters
            try:
                with st.spinner("Waiting for the API…"):
                    post = requests.post(
                        _api_url("/query"),
                        json=payload,
                        timeout=REQUEST_TIMEOUT,
                    )
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
            else:
                if post.status_code != 200:
                    st.error(f"POST /query failed: HTTP {post.status_code}\n\n{post.text}")
                else:
                    st.session_state.last_response = post.json()

    result = st.session_state.last_response
    if result is not None:
        answer = str(result.get("answer", ""))
        sources = result.get("sources") or []
        chunk_count = int(result.get("chunk_count", 0))

        _maybe_warn_empty(answer, chunk_count)

        reformulated_query = result.get("reformulated_query")
        if reformulated_query:
            st.info(f"Query was weak — automatically reformulated to:\n\n> *{reformulated_query}*")

        st.markdown("### Answer")
        st.info(answer)

        with st.expander(f"Sources ({chunk_count} chunks)", expanded=False):
            if not sources:
                st.caption("No source rows in the response.")
            else:
                r_scores = [float(s.get("rerank_score", 0.0)) for s in sources]
                normed = _normalize_rerank_scores(r_scores)
                for src, bar_val in zip(sources, normed):
                    company = src.get("company", "")
                    doc_type = src.get("doc_type", "")
                    fiscal_year = src.get("fiscal_year", "")
                    filing_date = src.get("filing_date", "")
                    page = src.get("page_number", "")
                    with st.container(border=True):
                        st.markdown(
                            f"**{company}** | {doc_type} | FY **{fiscal_year}** | "
                            f"filed **{filing_date}** | page **{page}**"
                        )
                        st.caption("Rerank score (normalized across this result set)")
                        st.progress(float(min(1.0, max(0.0, bar_val))))


if __name__ == "__main__":
    main()