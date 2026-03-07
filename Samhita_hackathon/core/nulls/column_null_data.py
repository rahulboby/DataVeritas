def getColumnsNullData(df, selected_cols, logic_mode):
    mask = df[selected_cols].isna().all(axis=1) if logic_mode == "AND" else df[selected_cols].isna().any(axis=1)
    match_count = mask.sum()
    masked_df = df[mask]
    return match_count, masked_df