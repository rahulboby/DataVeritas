import pandas as pd
import numpy as np
from typing import Optional, List


def compute_scores(df: pd.DataFrame, rules: Optional[list] = None) -> dict:
    """Compute overall DQ score and all sub-scores."""
    from core.score.overall_score import get_overall_score
    
    dq_score, null_score, completeness, uniqueness, outlier, consistency = get_overall_score(df, rules=rules)
    
    # Trust level
    if dq_score > 0.9:
        trust_level = "Excellent"
    elif dq_score > 0.7:
        trust_level = "Moderate"
    elif dq_score > 0.5:
        trust_level = "Low"
    else:
        trust_level = "Critical"
    
    return {
        "dq_score": round(float(dq_score), 4),
        "trust_level": trust_level,
        "null_score": round(float(null_score), 4),
        "completeness_score": round(float(completeness), 4),
        "uniqueness_score": round(float(uniqueness), 4),
        "outlier_score": round(float(outlier), 4),
        "consistency_score": round(float(consistency), 4),
        "weights": {
            "null": 0.1,
            "completeness": 0.3,
            "uniqueness": 0.3,
            "outlier": 0.1,
            "consistency": 0.2,
        }
    }


def compute_field_score(df: pd.DataFrame, column: str) -> dict:
    """Compute per-field DQ score."""
    from core.score.overall_field_score import get_overall_field_score
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")
    
    overall, null_s, unique_s, outlier_s = get_overall_field_score(df, column)
    
    return {
        "column": column,
        "overall_score": round(float(overall), 4),
        "null_score": round(float(null_s), 4),
        "unique_score": round(float(unique_s), 4),
        "outlier_score": round(float(outlier_s), 4),
    }


def compute_null_analysis(df: pd.DataFrame) -> dict:
    """Comprehensive null/completeness analysis."""
    from core.nulls.null_score import get_null_score
    from core.nulls.completeness_score import get_completeness_score
    
    total_nulls = int(df.isnull().sum().sum())
    total_cells = df.shape[0] * df.shape[1]
    
    # Per-column null stats
    null_per_col = df.isnull().sum()
    per_column = []
    for col in df.columns:
        count = int(null_per_col[col])
        per_column.append({
            "column": col,
            "null_count": count,
            "null_pct": round(count / len(df) * 100, 2) if len(df) > 0 else 0,
            "dtype": str(df[col].dtype),
        })
    
    # Sort by null count descending
    per_column.sort(key=lambda x: x["null_count"], reverse=True)
    
    # Row-level stats
    null_rows = int(df.isnull().any(axis=1).sum())
    complete_rows = len(df) - null_rows
    
    return {
        "null_score": round(float(get_null_score(df)), 4),
        "completeness_score": round(float(get_completeness_score(df)), 4),
        "total_nulls": total_nulls,
        "total_cells": total_cells,
        "null_pct": round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0,
        "null_rows": null_rows,
        "complete_rows": complete_rows,
        "null_rows_pct": round(null_rows / len(df) * 100, 2) if len(df) > 0 else 0,
        "per_column": per_column,
    }


def compute_duplicate_analysis(df: pd.DataFrame) -> dict:
    """Comprehensive duplicate analysis."""
    from core.cardinality.uniqueness_score import get_uniqueness_score
    
    total = len(df)
    dup_mask = df.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    unique_count = total - dup_count
    dedup_count = len(df.drop_duplicates())
    redundant = total - dedup_count
    
    return {
        "uniqueness_score": round(float(get_uniqueness_score(df)), 4),
        "total_rows": total,
        "unique_rows": unique_count,
        "duplicate_rows": dup_count,
        "deduplicated_count": dedup_count,
        "redundant_rows": redundant,
        "duplicate_pct": round(dup_count / total * 100, 2) if total > 0 else 0,
    }


def compute_outlier_analysis(df: pd.DataFrame) -> dict:
    """Comprehensive outlier analysis."""
    from core.outliers.outlier_score import get_outlier_score, get_column_stats
    
    score, outlier_df = get_outlier_score(df)
    _, _, _, _, numeric_cols, _, _ = get_column_stats(df)
    
    # Per-column outlier stats using IQR
    per_column = []
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 5:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((values < lower) | (values > upper)).sum())
        
        per_column.append({
            "column": col,
            "outlier_count": n_outliers,
            "outlier_pct": round(n_outliers / len(values) * 100, 2) if len(values) > 0 else 0,
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "min": round(float(values.min()), 2),
            "max": round(float(values.max()), 2),
            "mean": round(float(values.mean()), 2),
            "median": round(float(values.median()), 2),
        })
    
    per_column.sort(key=lambda x: x["outlier_count"], reverse=True)
    
    return {
        "outlier_score": round(float(score), 4),
        "total_outlier_rows": len(outlier_df) if outlier_df is not None else 0,
        "numeric_columns": len(numeric_cols),
        "columns_with_outliers": len([c for c in per_column if c["outlier_count"] > 0]),
        "per_column": per_column,
    }


def compute_consistency_analysis(df: pd.DataFrame, rules: Optional[list] = None) -> dict:
    """Consistency analysis with rules."""
    from core.consistency.consistency_score import get_consistency_score
    
    score, violation_df = get_consistency_score(df, rules=rules)
    
    violations_list = []
    if not violation_df.empty and 'violation_reason' in violation_df.columns:
        # Summarize violations
        reasons = violation_df['violation_reason'].value_counts()
        for reason, count in reasons.items():
            violations_list.append({
                "reason": str(reason),
                "count": int(count),
                "pct": round(int(count) / len(df) * 100, 2) if len(df) > 0 else 0,
            })
    
    return {
        "consistency_score": round(float(score), 4),
        "total_violations": len(violation_df),
        "violation_pct": round(len(violation_df) / len(df) * 100, 2) if len(df) > 0 else 0,
        "violations": violations_list,
    }


def compute_distribution_analysis(df: pd.DataFrame) -> dict:
    """Column type distribution and stats."""
    from core.value_distribution.columns_stats import get_column_stats
    
    counts, constant, datetime_cols, boolean, numeric, categorical, identifier = get_column_stats(df)
    
    column_details = []
    for col in df.columns:
        detail = {
            "column": col,
            "dtype": str(df[col].dtype),
            "unique_count": int(df[col].nunique()),
            "null_count": int(df[col].isnull().sum()),
        }
        
        if col in numeric:
            detail["type"] = "numeric"
            vals = df[col].dropna()
            if len(vals) > 0:
                detail["stats"] = {
                    "mean": round(float(vals.mean()), 4),
                    "median": round(float(vals.median()), 4),
                    "std": round(float(vals.std()), 4),
                    "min": round(float(vals.min()), 4),
                    "max": round(float(vals.max()), 4),
                }
        elif col in categorical:
            detail["type"] = "categorical"
            top = df[col].value_counts(dropna=True).head(10)
            detail["top_values"] = {str(k): int(v) for k, v in top.items()}
        elif col in datetime_cols:
            detail["type"] = "datetime"
        elif col in boolean:
            detail["type"] = "boolean"
        elif col in constant:
            detail["type"] = "constant"
        else:
            detail["type"] = "identifier"
        
        column_details.append(detail)
    
    return {
        "column_types": counts,
        "total_columns": len(df.columns),
        "columns": column_details,
    }
