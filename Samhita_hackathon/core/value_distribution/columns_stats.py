def get_column_stats(df):
    """Categorizes columns with strict 40-category limit for categorical fields."""
    all_cols = df.columns.tolist()
    
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    remaining = [col for col in all_cols if col not in constant_cols]

    datetime_cols = df[remaining].select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    remaining = [col for col in remaining if col not in datetime_cols]

    boolean_cols = df[remaining].select_dtypes(include=['bool']).columns.tolist()
    remaining = [col for col in remaining if col not in boolean_cols]

    numeric_cols = df[remaining].select_dtypes(include=['number']).columns.tolist()
    remaining = [col for col in remaining if col not in numeric_cols]

    # STIPULATION: Only columns with <= 40 unique values are Categorical
    categorical_cols = [col for col in remaining if df[col].nunique() <= (df.shape[0]/4)]
    remaining = [col for col in remaining if col not in categorical_cols]

    identifier_cols = [col for col in remaining]
    
    counts = {
        "Numeric": len(numeric_cols),
        "Categorical": len(categorical_cols),
        "Datetime": len(datetime_cols),
        "Boolean": len(boolean_cols),
        "Constant": len(constant_cols),
        "Identifier": len(identifier_cols)
    }
    
    return counts, constant_cols, datetime_cols, boolean_cols, numeric_cols, categorical_cols, identifier_cols
