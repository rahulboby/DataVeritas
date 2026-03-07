def getOverallFieldScore(df, selected_column, ):
    from core.outliers.outlier_score import getOutlierScore
    null_score_field = df[selected_column].notnull().mean() if len(df) > 0 else 0
    unique_score_field = df[selected_column].nunique() / len(df) if len(df) > 0 else 0
    outlier_score_field = getOutlierScore(df[[selected_column]])[0]
    null_w_field = 0.4
    uniqueness_w_field = 0.4    
    outlier_w_field = 0.2
    overall_field_score = (null_score_field * null_w_field
        + unique_score_field * uniqueness_w_field
        + outlier_score_field * outlier_w_field)
    return overall_field_score, null_score_field, unique_score_field, outlier_score_field
