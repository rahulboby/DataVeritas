import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_dynamic_weights(df, rules=None):
    """
    Compute entropy-based dynamic weights for the trust score dimensions.
    
    Instead of fixed weights, this uses the entropy (uncertainty) of each
    quality dimension to assign importance dynamically.
    
    Higher entropy = more variation = needs more attention = higher weight.
    
    Returns:
        dict with keys: weights, static_score, dynamic_score, comparison
    """
    from core.nulls.null_score import get_null_score
    from core.nulls.completeness_score import get_completeness_score
    from core.cardinality.uniqueness_score import get_uniqueness_score
    from core.outliers.outlier_score import get_outlier_score
    from core.consistency.consistency_score import get_consistency_score

    # Get all sub-scores
    null_score = get_null_score(df)
    completeness_score = get_completeness_score(df)
    uniqueness_score = get_uniqueness_score(df)
    outlier_score, _ = get_outlier_score(df)
    violation_score, _ = get_consistency_score(df, rules=rules)

    scores = {
        "null": null_score,
        "completeness": completeness_score,
        "uniqueness": uniqueness_score,
        "outlier": outlier_score,
        "consistency": violation_score,
    }

    # Compute per-column entropy contributions for each dimension
    entropies = {}
    
    # Null entropy: how spread out null values are across columns
    null_rates = df.isnull().mean()
    entropies["null"] = _safe_entropy(null_rates.values)
    
    # Completeness entropy based on per-column fill rates
    fill_rates = 1 - null_rates
    entropies["completeness"] = _safe_entropy(fill_rates.values)
    
    # Uniqueness entropy: variation in duplication rates per column
    dup_rates = []
    for col in df.columns:
        total = len(df)
        if total > 0:
            dup_rates.append(df[col].duplicated().mean())
        else:
            dup_rates.append(0)
    entropies["uniqueness"] = _safe_entropy(np.array(dup_rates))
    
    # Outlier entropy: only for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        skewness_vals = df[numeric_cols].skew().abs().fillna(0).values
        entropies["outlier"] = _safe_entropy(skewness_vals / (skewness_vals.sum() + 1e-10))
    else:
        entropies["outlier"] = 0.0

    # Consistency entropy: uniform assumption (no per-column breakdown easily available)
    entropies["consistency"] = _safe_entropy(np.array([violation_score, 1 - violation_score]))

    # Normalize entropies to get weights
    total_entropy = sum(entropies.values())
    if total_entropy == 0:
        # Fallback to equal weights
        dynamic_weights = {k: 0.2 for k in entropies}
    else:
        dynamic_weights = {k: v / total_entropy for k, v in entropies.items()}

    # Static weights (original)
    static_weights = {
        "null": 0.1,
        "completeness": 0.3,
        "uniqueness": 0.3,
        "outlier": 0.1,
        "consistency": 0.2,
    }

    # Compute both scores
    static_score = sum(scores[k] * static_weights[k] for k in scores)
    dynamic_score = sum(scores[k] * dynamic_weights[k] for k in scores)

    return {
        "scores": scores,
        "static_weights": static_weights,
        "dynamic_weights": dynamic_weights,
        "static_score": round(static_score, 4),
        "dynamic_score": round(dynamic_score, 4),
        "entropies": {k: round(v, 4) for k, v in entropies.items()},
    }


def _safe_entropy(values):
    """Calculate Shannon entropy from an array of probabilities/proportions."""
    values = np.array(values, dtype=float)
    values = values[values > 0]  # Remove zeros
    if len(values) == 0:
        return 0.0
    # Normalize to probability distribution
    values = values / values.sum()
    return float(scipy_stats.entropy(values))
