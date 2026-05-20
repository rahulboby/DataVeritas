import hashlib
import os

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = (
    "You are a concise data quality analyst inside DataVeritas. "
    "Write executive-quality markdown reports using only the supplied metrics. "
    "Do not invent fields, causes, or counts."
)


def render_trust_score_report(
    df: pd.DataFrame,
    score_results,
    outlier_df: pd.DataFrame | None = None,
    violation_df: pd.DataFrame | None = None,
):
    """
    Render the Trust Score NLP report expander.
    The LLM receives summary metrics only, never raw dataset rows.
    """
    with st.expander("NLP Trust Score Report", expanded=False):
        st.caption(
            "Generate a concise report from the trust-score summary, missingness, duplicates, "
            "outliers, consistency results, and field-level risk signals."
        )

        context = build_trust_score_context(df, score_results, outlier_df, violation_df)
        report_key = "trust_score_nlp_report"
        context_hash_key = "trust_score_nlp_context_hash"
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()

        if st.session_state.get(context_hash_key) != context_hash:
            st.session_state[context_hash_key] = context_hash
            st.session_state[report_key] = None

        col1, col2 = st.columns([1, 4])

        with col1:
            generate = st.button(
                "Generate Report",
                type="primary",
                key="generate_trust_score_nlp_report",
                width="stretch",
            )

        with col2:
            st.caption(f"Provider: Groq | Model: {DEFAULT_GROQ_MODEL}")

        if generate:
            try:
                with st.spinner("Generating trust score report..."):
                    st.session_state[report_key] = generate_trust_score_report(context)
            except ValueError as exc:
                st.warning(str(exc))
                st.markdown(build_local_report_fallback(context))
            except Exception as exc:
                st.error(f"Failed to generate NLP report: {exc}")
                st.markdown(build_local_report_fallback(context))

        if st.session_state.get(report_key):
            st.markdown(st.session_state[report_key])
        else:
            st.info("Open this section and click Generate Report to create the NLP summary.")

        with st.expander("Report Context", expanded=False):
            st.code(context, language="text")


def build_trust_score_context(
    df: pd.DataFrame,
    score_results,
    outlier_df: pd.DataFrame | None = None,
    violation_df: pd.DataFrame | None = None,
) -> str:
    dq_score, null_score, completeness_score, uniqueness_score, outlier_score, violation_score = score_results

    nrows, ncols = df.shape
    total_cells = nrows * ncols
    null_counts = df.isna().sum()
    total_nulls = int(null_counts.sum())
    rows_with_nulls = int(df.isna().any(axis=1).sum())
    duplicate_rows = int(df.duplicated(keep=False).sum())
    redundant_duplicate_rows = int(df.duplicated(keep="first").sum())
    outlier_rows = len(outlier_df) if isinstance(outlier_df, pd.DataFrame) else 0
    consistency_rows = len(violation_df) if isinstance(violation_df, pd.DataFrame) else 0

    top_missing = build_top_missing_fields(df, null_counts)
    cardinality_summary = build_cardinality_summary(df)
    outlier_summary = build_numeric_outlier_summary(df)
    pii_columns = find_potential_pii_columns(df)

    lines = [
        f"Dataset rows: {nrows:,}",
        f"Dataset columns: {ncols:,}",
        f"Total cells: {total_cells:,}",
        "",
        "Trust score components:",
        f"- Overall trust score: {dq_score:.2%}",
        f"- Non-null record score: {null_score:.2%}",
        f"- Completeness score: {completeness_score:.2%}",
        f"- Uniqueness score: {uniqueness_score:.2%}",
        f"- Outlier score: {outlier_score:.2%}",
        f"- Consistency score: {violation_score:.2%}",
        "",
        "Detected issue counts:",
        f"- Total null cells: {total_nulls:,}",
        f"- Rows with at least one null: {rows_with_nulls:,}",
        f"- Rows involved in exact duplicates: {duplicate_rows:,}",
        f"- Redundant exact duplicate rows: {redundant_duplicate_rows:,}",
        f"- Outlier rows detected by anomaly module: {outlier_rows:,}",
        f"- Consistency violation rows: {consistency_rows:,}",
    ]

    append_section(lines, "Top missing fields", top_missing)
    append_section(lines, "Cardinality signals", cardinality_summary)
    append_section(lines, "Top numeric IQR outlier fields", outlier_summary)
    append_section(lines, "Potential PII fields", pii_columns)

    return "\n".join(lines)


def build_top_missing_fields(df: pd.DataFrame, null_counts: pd.Series, limit: int = 8) -> list[str]:
    nrows = len(df)
    missing = null_counts[null_counts > 0].sort_values(ascending=False).head(limit)
    return [
        f"{col}: {int(count):,} nulls ({count / nrows:.2%})"
        for col, count in missing.items()
    ]


def build_cardinality_summary(df: pd.DataFrame, limit: int = 8) -> list[str]:
    if df.empty:
        return []

    nrows = len(df)
    rows = []
    constant_cols = []
    high_unique_cols = []

    for col in df.columns:
        unique_count = int(df[col].nunique(dropna=False))
        ratio = unique_count / nrows if nrows else 0

        if unique_count <= 1:
            constant_cols.append(col)
        elif ratio >= 0.90:
            high_unique_cols.append(f"{col} ({ratio:.1%} unique)")

    if constant_cols:
        rows.append("Constant fields: " + ", ".join(constant_cols[:limit]))
    if high_unique_cols:
        rows.append("High-uniqueness fields: " + ", ".join(high_unique_cols[:limit]))

    return rows


def build_numeric_outlier_summary(df: pd.DataFrame, limit: int = 8) -> list[str]:
    numeric_df = df.select_dtypes(include=[np.number], exclude=["bool"])
    rows = []

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())

        if count > 0:
            rows.append((col, count, count / len(series)))

    rows.sort(key=lambda item: item[1], reverse=True)
    return [f"{col}: {count:,} IQR outliers ({pct:.2%})" for col, count, pct in rows[:limit]]


def find_potential_pii_columns(df: pd.DataFrame, limit: int = 10) -> list[str]:
    pii_keywords = [
        "email",
        "phone",
        "ssn",
        "address",
        "name",
        "dob",
        "birth",
        "passport",
        "license",
        "credit",
    ]
    return [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in pii_keywords)
    ][:limit]


def append_section(lines: list[str], title: str, values: list[str]):
    lines.append("")
    lines.append(f"{title}:")
    if values:
        for value in values:
            lines.append(f"- {value}")
    else:
        lines.append("- None detected")


def generate_trust_score_report(context: str) -> str:
    client = get_groq_client()
    prompt = (
        "Using the DataVeritas trust score context below, write a polished report of about 200 words. "
        "Use markdown with these headings: **Trust Score Summary**, **Issues Found**, "
        "and **Recommended Fixes**. Include concise bullet points under Issues and Fixes. "
        "Reference exact scores, counts, and fields from the context. Stay under 220 words.\n\n"
        f"Context:\n{context}"
    )

    response = client.chat.completions.create(
        model=DEFAULT_GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
        max_completion_tokens=450,
        top_p=1,
    )

    return response.choices[0].message.content.strip()


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "Groq API key not found. Set GROQ_API_KEY as an environment variable "
            "or add it to .streamlit/secrets.toml."
        )

    from groq import Groq

    return Groq(api_key=api_key)


def build_local_report_fallback(context: str) -> str:
    return (
        "**Local Fallback Summary**\n\n"
        "The NLP provider was unavailable, so the full language report could not be generated. "
        "The computed DataVeritas context is still available below in **Report Context**. "
        "Configure `GROQ_API_KEY` to generate the 200-word markdown report from these metrics."
    )
