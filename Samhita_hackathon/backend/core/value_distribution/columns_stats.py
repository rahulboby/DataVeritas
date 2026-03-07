def get_column_stats(df):
    """
    Categorizes columns with type detection.
    
    Returns:
        Tuple of (counts_dict, constant_cols, datetime_cols, boolean_cols, 
                  numeric_cols, categorical_cols, identifier_cols)
    """
    all_cols = df.columns.tolist()

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    remaining = [col for col in all_cols if col not in constant_cols]

    datetime_cols = df[remaining].select_dtypes(include=['datetime', 'datetimetz']).columns.tolist() if remaining else []
    remaining = [col for col in remaining if col not in datetime_cols]

    boolean_cols = df[remaining].select_dtypes(include=['bool']).columns.tolist() if remaining else []
    remaining = [col for col in remaining if col not in boolean_cols]

    numeric_cols = df[remaining].select_dtypes(include=['number']).columns.tolist() if remaining else []
    remaining = [col for col in remaining if col not in numeric_cols]

    # Only columns with <= N/4 unique values are Categorical
    categorical_cols = [col for col in remaining if df[col].nunique() <= (df.shape[0] / 4)]
    remaining = [col for col in remaining if col not in categorical_cols]

    identifier_cols = list(remaining)

    counts = {
        "Numeric": len(numeric_cols),
        "Categorical": len(categorical_cols),
        "Datetime": len(datetime_cols),
        "Boolean": len(boolean_cols),
        "Constant": len(constant_cols),
        "Identifier": len(identifier_cols)
    }

    return counts, constant_cols, datetime_cols, boolean_cols, numeric_cols, categorical_cols, identifier_cols
