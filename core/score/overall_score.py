def getOverallScore(df, null_w=0.1, completeness_w=0.3, uniqueness_w=0.3, outlier_w=0.1, violation_w=0.2):
    
    from core.nulls.null_score import getNullScore
    from core.nulls.completeness_score import getCompletenessScore
    from core.cardinality.uniqueness_score import getUniquenessScore
    from core.consistency.consistency_score_and_df import getConsistencyScore
    from core.outliers.outlier_score import getOutlierScore
    
    null_score = getNullScore(df)
    completeness_score = getCompletenessScore(df)
    uniqueness_score = getUniquenessScore(df)
    outlier_score, _ = getOutlierScore(df)
    violation_score, _ = getConsistencyScore(df)

    total_weight = (
        null_w
        + completeness_w
        + uniqueness_w
        + outlier_w
        + violation_w
    )

    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")
    
    # Final Aggregate Score
    weighted_sum = (null_score * null_w
            + completeness_score * completeness_w 
            + uniqueness_score * uniqueness_w
            + outlier_score * outlier_w
            + violation_score * violation_w) # SCORE
    
    dq_score = weighted_sum / total_weight

    return dq_score, null_score, completeness_score, uniqueness_score, outlier_score, violation_score
