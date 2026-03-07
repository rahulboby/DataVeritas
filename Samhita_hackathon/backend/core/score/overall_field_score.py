def get_overall_field_score(df, selected_column):
    """
    Calculate per-field DQ score.
    
    Weights:
        - Null (completeness): 40%
        - Uniqueness: 40%
        - Outlier-free: 20%
    
    Returns:
        Tuple of (overall_field_score, null_score_field, unique_score_field, outlier_score_field)
    """
    from core.outliers.outlier_score import get_outlier_score

    null_score_field = df[selected_column].notnull().mean() if len(df) > 0 else 0
    unique_score_field = df[selected_column].nunique() / len(df) if len(df) > 0 else 0
    outlier_score_field = get_outlier_score(df[[selected_column]])[0]

    null_w_field = 0.4
    uniqueness_w_field = 0.4
    outlier_w_field = 0.2

    overall_field_score = (
        null_score_field * null_w_field
        + unique_score_field * uniqueness_w_field
        + outlier_score_field * outlier_w_field
    )

    return overall_field_score, null_score_field, unique_score_field, outlier_score_field
