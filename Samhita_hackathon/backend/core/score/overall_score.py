def get_overall_score(df, rules=None):
    """
    Calculate the overall Data Quality score as a weighted average of 5 sub-scores.
    
    Weights:
        - Null score: 10%
        - Completeness: 30%
        - Uniqueness: 30%
        - Outlier: 10%
        - Consistency: 20%
    
    Args:
        df: Input DataFrame
        rules: Optional consistency rules list
    
    Returns:
        Tuple of (dq_score, null_score, completeness_score, 
                  uniqueness_score, outlier_score, violation_score)
    """
    from core.nulls.null_score import get_null_score
    from core.nulls.completeness_score import get_completeness_score
    from core.cardinality.uniqueness_score import get_uniqueness_score
    from core.consistency.consistency_score import get_consistency_score
    from core.outliers.outlier_score import get_outlier_score

    null_score = get_null_score(df)
    completeness_score = get_completeness_score(df)
    uniqueness_score = get_uniqueness_score(df)
    outlier_score, _ = get_outlier_score(df)
    violation_score, _ = get_consistency_score(df, rules=rules)

    null_w = 0.1
    completeness_w = 0.3
    uniqueness_w = 0.3
    outlier_w = 0.1
    violation_w = 0.2

    dq_score = (
        null_score * null_w
        + completeness_score * completeness_w
        + uniqueness_score * uniqueness_w
        + outlier_score * outlier_w
        + violation_score * violation_w
    )

    return dq_score, null_score, completeness_score, uniqueness_score, outlier_score, violation_score
