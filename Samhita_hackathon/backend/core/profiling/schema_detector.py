import pandas as pd
import numpy as np


def detect_schema(df):
    """
    Enhanced schema detection and dataset profiling.
    
    Detects:
    - Primary key candidates
    - Column types (numeric, categorical, datetime, boolean, identifier, constant)
    - High cardinality columns
    - Per-column statistics
    
    Returns:
        dict with schema profile
    """
    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "columns": [],
        "primary_key_candidates": [],
        "type_summary": {},
    }
    
    type_counts = {
        "numeric": 0, "categorical": 0, "datetime": 0,
        "boolean": 0, "identifier": 0, "constant": 0
    }
    
    for col in df.columns:
        col_info = _profile_column(df, col)
        profile["columns"].append(col_info)
        type_counts[col_info["detected_type"]] += 1
        
        # Primary key candidate: unique, no nulls
        if col_info["null_count"] == 0 and col_info["unique_count"] == len(df):
            profile["primary_key_candidates"].append(col)
    
    profile["type_summary"] = type_counts
    return profile


def _profile_column(df, col):
    """Generate a detailed profile for a single column."""
    series = df[col]
    total = len(series)
    null_count = int(series.isnull().sum())
    unique_count = int(series.nunique(dropna=True))
    
    col_info = {
        "name": col,
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_pct": round(null_count / total * 100, 2) if total > 0 else 0,
        "unique_count": unique_count,
        "cardinality_ratio": round(unique_count / total, 4) if total > 0 else 0,
    }
    
    # Detect type
    if series.nunique() <= 1:
        col_info["detected_type"] = "constant"
        col_info["constant_value"] = str(series.mode().iloc[0]) if len(series.mode()) > 0 else None
    elif pd.api.types.is_bool_dtype(series):
        col_info["detected_type"] = "boolean"
    elif pd.api.types.is_datetime64_any_dtype(series):
        col_info["detected_type"] = "datetime"
        non_null = series.dropna()
        if len(non_null) > 0:
            col_info["min"] = str(non_null.min())
            col_info["max"] = str(non_null.max())
    elif pd.api.types.is_numeric_dtype(series):
        col_info["detected_type"] = "numeric"
        non_null = series.dropna()
        if len(non_null) > 0:
            col_info["min"] = float(non_null.min())
            col_info["max"] = float(non_null.max())
            col_info["mean"] = round(float(non_null.mean()), 4)
            col_info["median"] = round(float(non_null.median()), 4)
            col_info["std"] = round(float(non_null.std()), 4)
            col_info["skewness"] = round(float(non_null.skew()), 4)
            col_info["kurtosis"] = round(float(non_null.kurtosis()), 4)
    elif series.dtype == 'object':
        cardinality_ratio = unique_count / total if total > 0 else 0
        if cardinality_ratio > 0.9:
            col_info["detected_type"] = "identifier"
        elif unique_count <= (total / 4):
            col_info["detected_type"] = "categorical"
            # Top values
            top_vals = series.value_counts(dropna=True).head(10)
            col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
        else:
            col_info["detected_type"] = "categorical"
            top_vals = series.value_counts(dropna=True).head(10)
            col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
    else:
        col_info["detected_type"] = "identifier"
    
    # Sample values
    non_null_sample = series.dropna().head(5)
    col_info["sample_values"] = [str(v) for v in non_null_sample.values]
    
    return col_info
