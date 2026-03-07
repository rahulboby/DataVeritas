"""
DataVeritas AI Assistant — Core LLM Logic

This module provides:
1. Insight extraction from the existing DataVeritas pipeline outputs
2. Groq LLM client with streaming support
3. Automatic AI summary generation

The LLM never sees raw data — only structured insights computed by the pipeline.
"""

import pandas as pd
import numpy as np
import streamlit as st
from groq import Groq

# ==================== CONFIGURATION ====================

# Groq API key — set via environment variable GROQ_API_KEY
# or via Streamlit secrets (st.secrets["GROQ_API_KEY"])
# The Groq() client auto-reads GROQ_API_KEY from env by default.

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a data quality analyst working inside the DataVeritas platform. "
    "You explain dataset quality issues, trust scores, drift detection results, "
    "and privacy risks clearly. "
    "Only answer using the provided dataset insights. Do not invent data."
)


# ==================== STEP 1: INSIGHT EXTRACTION ====================

def extract_dataset_insights(df: pd.DataFrame) -> dict:
    """
    Collect all analysis results from the existing pipeline into a single
    structured dictionary.  Only calls existing functions — never modifies them.

    The returned dict contains:
      - Dataset-level metrics (trust_score, duplicate_records, etc.)
      - A "columns" dict with per-column statistics
    """
    from core.score.overall_score import getOverallScore
    from core.nulls.null_score import getNullScore
    from core.nulls.completeness_score import getCompletenessScore
    from core.cardinality.uniqueness_score import getUniquenessScore
    from core.outliers.outlier_score import getOutlierScore
    from core.consistency.consistency_score_and_df import getConsistencyScore
    from core.value_distribution.columns_stats import get_column_stats

    # --- Overall scores ---
    dq_score, null_score, completeness_score, uniqueness_score, outlier_score, violation_score = getOverallScore(df)

    # --- Column type breakdown ---
    col_counts, constant_cols, datetime_cols, boolean_cols, numeric_cols, categorical_cols, identifier_cols = get_column_stats(df)

    # --- Missing values per column ---
    null_counts = df.isnull().sum()
    total_rows = len(df)
    columns_with_missing = {
        col: f"{(count / total_rows * 100):.1f}%"
        for col, count in null_counts.items() if count > 0
    }

    # --- Duplicates ---
    dup_mask = df.duplicated(keep=False)
    duplicate_records = int(dup_mask.sum())

    # --- High-uniqueness / cardinality ---
    cardinality_data = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        ratio = round(unique_count / total_rows, 4) if total_rows > 0 else 0
        cardinality_data.append({"column": col, "unique_count": unique_count, "ratio": ratio})

    high_uniqueness_columns = [c["column"] for c in cardinality_data if c["ratio"] >= 0.90]
    constant_columns = [c["column"] for c in cardinality_data if c["unique_count"] <= 1]

    # --- Consistency violations ---
    _, violation_df = getConsistencyScore(df)
    consistency_violations = []
    if isinstance(violation_df, pd.DataFrame) and not violation_df.empty:
        if "violation_reason" in violation_df.columns:
            consistency_violations = violation_df["violation_reason"].unique().tolist()

    # --- Outlier details ---
    _, outlier_df = getOutlierScore(df)
    outlier_row_count = 0
    if isinstance(outlier_df, pd.DataFrame) and not outlier_df.empty:
        outlier_row_count = len(outlier_df)

    # --- Privacy risk heuristic (columns that look like PII) ---
    pii_keywords = ["email", "phone", "ssn", "address", "name", "dob", "birth", "passport", "license", "credit"]
    privacy_risk_columns = [
        col for col in df.columns
        if any(kw in col.lower() for kw in pii_keywords)
    ]

    # ================================================================
    # STEP 2 — Column-Level Features
    # ================================================================
    columns_info = {}
    numeric_set = set(numeric_cols)
    categorical_set = set(categorical_cols)

    for col in df.columns:
        col_null_count = int(null_counts.get(col, 0))
        col_unique = int(df[col].nunique(dropna=False))

        col_stats = {
            "missing_values": col_null_count,
            "missing_pct": round(col_null_count / total_rows * 100, 2) if total_rows > 0 else 0,
            "uniqueness_ratio": round(col_unique / total_rows, 4) if total_rows > 0 else 0,
            "cardinality": col_unique,
            "datatype": str(df[col].dtype),
        }

        # Numeric columns — add descriptive statistics
        if col in numeric_set:
            series = df[col].dropna()
            if len(series) > 0:
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                iqr_outliers = int(((series < lower_fence) | (series > upper_fence)).sum())

                col_stats.update({
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr_outlier_count": iqr_outliers,
                })

        # Categorical columns — add top value
        if col in categorical_set:
            vc = df[col].value_counts(dropna=True)
            if len(vc) > 0:
                col_stats["top_value"] = str(vc.index[0])
                col_stats["top_frequency"] = int(vc.iloc[0])

        # Privacy flag
        col_stats["is_privacy_risk"] = col in privacy_risk_columns

        columns_info[col] = col_stats

    # ================================================================
    # Build final insights dict
    # ================================================================
    insights = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "trust_score": round(dq_score * 100, 1),
        "null_score": round(null_score * 100, 1),
        "completeness_score": round(completeness_score * 100, 1),
        "uniqueness_score": round(uniqueness_score * 100, 1),
        "outlier_score": round(outlier_score * 100, 1),
        "consistency_score": round(violation_score * 100, 1),
        "columns_with_missing_values": columns_with_missing,
        "duplicate_records": duplicate_records,
        "drift_detected": None,          # placeholder — populated by drift detector when available
        "drift_columns": [],             # placeholder
        "high_uniqueness_columns": high_uniqueness_columns,
        "constant_columns": constant_columns,
        "consistency_rule_violations": consistency_violations,
        "outlier_row_count": outlier_row_count,
        "privacy_risk_columns": privacy_risk_columns,
        "column_type_breakdown": col_counts,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "columns": columns_info,          # per-column detailed stats
    }

    return insights


def insights_to_context(insights: dict) -> str:
    """
    Convert the structured insights dictionary into a human-readable
    context string suitable for inclusion in an LLM prompt.
    """
    lines = []

    lines.append(f"Dataset Size: {insights['total_rows']:,} rows × {insights['total_columns']} columns")
    lines.append("")
    lines.append(f"Overall Dataset Trust Score: {insights['trust_score']}%")
    lines.append(f"  - Null Score (rows with no nulls): {insights['null_score']}%")
    lines.append(f"  - Completeness Score (non-null cells): {insights['completeness_score']}%")
    lines.append(f"  - Uniqueness Score: {insights['uniqueness_score']}%")
    lines.append(f"  - Outlier Score (non-outlier rows): {insights['outlier_score']}%")
    lines.append(f"  - Consistency Score (rule compliance): {insights['consistency_score']}%")
    lines.append("")

    # Missing values
    if insights["columns_with_missing_values"]:
        lines.append("Missing Values:")
        for col, pct in insights["columns_with_missing_values"].items():
            lines.append(f"  {col}: {pct}")
    else:
        lines.append("Missing Values: None detected")
    lines.append("")

    # Duplicates
    lines.append(f"Duplicate Records: {insights['duplicate_records']:,}")
    lines.append("")

    # Drift
    drift = insights.get("drift_detected")
    if drift is True:
        lines.append("Drift Detected: Yes")
        drift_cols = insights.get("drift_columns", [])
        if drift_cols:
            lines.append(f"  Affected columns: {', '.join(drift_cols)}")
    elif drift is False:
        lines.append("Drift Detected: No")
    else:
        lines.append("Drift Detected: Not evaluated")
    lines.append("")

    # High uniqueness
    if insights["high_uniqueness_columns"]:
        lines.append("Columns with High Uniqueness (≥90%):")
        lines.append("  " + ", ".join(insights["high_uniqueness_columns"]))
    lines.append("")

    # Constant columns
    if insights["constant_columns"]:
        lines.append("Constant Columns (≤1 unique value, zero information):")
        lines.append("  " + ", ".join(insights["constant_columns"]))
    lines.append("")

    # Consistency violations
    if insights["consistency_rule_violations"]:
        lines.append("Consistency Rule Violations:")
        for v in insights["consistency_rule_violations"][:15]:  # cap for context window
            lines.append(f"  - {v}")
    else:
        lines.append("Consistency Rule Violations: None detected")
    lines.append("")

    # Outliers
    lines.append(f"Outlier Rows Detected (Isolation Forest): {insights['outlier_row_count']:,}")
    lines.append("")

    # Privacy risk
    if insights["privacy_risk_columns"]:
        lines.append("Privacy Risk Columns (potential PII):")
        lines.append("  " + ", ".join(insights["privacy_risk_columns"]))
    else:
        lines.append("Privacy Risk Columns: None detected")
    lines.append("")

    # Column types
    lines.append("Column Type Breakdown:")
    for ctype, count in insights["column_type_breakdown"].items():
        if count > 0:
            lines.append(f"  {ctype}: {count}")

    # ================================================================
    # Column Analysis — per-column detail
    # ================================================================
    columns = insights.get("columns", {})
    if columns:
        lines.append("")
        lines.append("=" * 50)
        lines.append("Column Analysis:")
        lines.append("=" * 50)

        # Sort: worst (highest missing %) first for relevance
        sorted_cols = sorted(
            columns.items(),
            key=lambda kv: kv[1].get("missing_pct", 0),
            reverse=True,
        )

        for col_name, stats in sorted_cols:
            lines.append("")
            lines.append(f"Column: {col_name}")
            lines.append(f"  Datatype: {stats['datatype']}")
            lines.append(f"  Missing Values: {stats['missing_values']:,} ({stats['missing_pct']}%)")
            lines.append(f"  Cardinality: {stats['cardinality']:,}")
            lines.append(f"  Uniqueness Ratio: {stats['uniqueness_ratio']}")

            # Numeric extras
            if "mean" in stats:
                lines.append(f"  Mean: {stats['mean']}  |  Std: {stats['std']}")
                lines.append(f"  Min: {stats['min']}  |  Max: {stats['max']}")
                lines.append(f"  Q1: {stats['q1']}  |  Q3: {stats['q3']}")
                lines.append(f"  IQR Outliers: {stats['iqr_outlier_count']:,}")

            # Categorical extras
            if "top_value" in stats:
                lines.append(f"  Top Value: {stats['top_value']} (freq: {stats['top_frequency']:,})")

            # Privacy flag
            if stats.get("is_privacy_risk"):
                lines.append("  ⚠ Flagged as potential PII")

    return "\n".join(lines)


def build_llm_context(insights: dict) -> str:
    """
    Alias for insights_to_context — converts the dataset_insights dict
    into a readable context string for LLM consumption.

    This function does NOT call the LLM. It only prepares the data.
    """
    return insights_to_context(insights)


# ==================== STEP 2 & 3: GROQ LLM CLIENT (STREAMING) ====================

def _get_groq_client():
    """
    Create and return a Groq client.
    Reads GROQ_API_KEY from environment automatically.
    Falls back to Streamlit secrets if available.
    """
    import os
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ValueError(
            "Groq API key not found. "
            "Set the GROQ_API_KEY environment variable or add it to .streamlit/secrets.toml"
        )

    return Groq(api_key=api_key)


def ask_llm(question: str, insights_context: str) -> str:
    """
    Send the dataset insights context + user question to the Groq LLM
    with streaming enabled.  Returns the full response text.
    """
    client = _get_groq_client()

    user_message = (
        "Here are the dataset insights from DataVeritas:\n\n"
        f"{insights_context}\n\n"
        f"User Question: {question}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )

    # Iterate over streamed chunks and build the full response
    buffer = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            buffer += delta.content

    return buffer


def ask_llm_stream(question: str, insights_context: str):
    """
    Generator version of ask_llm — yields token strings one at a time.
    Designed for use with st.write_stream().
    """
    client = _get_groq_client()

    user_message = (
        "Here are the dataset insights from DataVeritas:\n\n"
        f"{insights_context}\n\n"
        f"User Question: {question}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ==================== STEP 5: AUTOMATIC AI SUMMARY ====================

def generate_ai_summary(insights_context: str) -> str:
    """
    Ask the LLM to produce a concise dataset health summary,
    major risks, and recommended actions.
    """
    prompt = (
        "Based on the dataset insights below, generate a concise report with exactly three sections:\n\n"
        "1. **Dataset Health Summary** — A 2-3 sentence overview of the dataset's overall quality.\n"
        "2. **Major Risks** — bullet list of the top issues affecting data reliability.\n"
        "3. **Recommended Actions** — bullet list of prioritized steps to improve data quality.\n\n"
        "Be specific and reference actual numbers from the insights."
    )
    return ask_llm(prompt, insights_context)
