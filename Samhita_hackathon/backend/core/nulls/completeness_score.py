def get_completeness_score(df):
    """
    Calculate cell-level completeness: ratio of non-null cells to total cells.
    Returns a float between 0 and 1.
    """
    total_nulls = df.isnull().sum().sum()
    nrows = df.shape[0]
    ncols = df.shape[1]
    total_cells = nrows * ncols
    if total_cells == 0:
        return 0.0
    return float((total_cells - total_nulls) / total_cells)
