def get_uniqueness_score(df):
    """
    Calculate uniqueness score: ratio of deduplicated rows to total rows.
    
    Pure unique rows + first occurrence of each duplicate group, divided by total.
    Returns a float between 0 and 1.
    """
    total_rows = df.shape[0]
    if total_rows == 0:
        return 0.0

    # Rows involved in duplication
    dup_mask = df.duplicated(keep=False)
    dup_count = int(dup_mask.sum())

    # Rows that are 100% unique
    pure_unique_count = total_rows - dup_count

    # After dedup (keeping first of each group)
    deduplicated_df = df.drop_duplicates(keep='first')
    deduplicated_count = deduplicated_df.shape[0]

    # Unique rows that have duplicates (first occurrence of each group)
    unique_with_duplicates = deduplicated_count - pure_unique_count

    return (pure_unique_count + unique_with_duplicates) / total_rows
