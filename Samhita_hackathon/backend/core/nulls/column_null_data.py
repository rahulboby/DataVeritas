def get_columns_null_data(df, selected_cols, logic_mode="OR"):
    """
    Get rows with nulls in the specified columns.
    
    Args:
        df: Input DataFrame
        selected_cols: List of column names to check
        logic_mode: "AND" (all columns null) or "OR" (any column null)
    
    Returns:
        Tuple of (match_count, masked_df)
    """
    if logic_mode == "AND":
        mask = df[selected_cols].isna().all(axis=1)
    else:
        mask = df[selected_cols].isna().any(axis=1)
    match_count = int(mask.sum())
    masked_df = df[mask]
    return match_count, masked_df
