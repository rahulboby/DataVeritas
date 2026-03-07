def get_null_score(df):
    """
    Calculate the ratio of fully non-null rows to total rows.
    Returns a float between 0 and 1.
    """
    total_rows = df.shape[0]
    if total_rows == 0:
        return 0.0
    return len(df.dropna()) / total_rows
