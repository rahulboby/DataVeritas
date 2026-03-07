def getUniquenessScore(df):
    # Temp Debugger:
    print(" --- Fetching Uniqueness Score --- ")
    
    # --- 1. CALCULATIONS ---
    total_rows = df.shape[0]
    
    # Rows involved in duplication (e.g., the 1,600)
    dup_mask = df.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    
    # Rows that are 100% unique (e.g., 18,400)
    pure_unique_count = total_rows - dup_count
    
    # Rows to remove (e.g., 1,200)
    
    #Filter out duplicates from [name, cid, email, phone]
    deduplicated_df_columnset =  df.drop_duplicates(keep = 'first')

    deduplicated_count = deduplicated_df_columnset.shape[0]
    redundant_rows_to_remove = total_rows - deduplicated_count
    
    # Unique rows that have duplicates (e.g., 400)
    unique_with_duplicates = deduplicated_count - pure_unique_count

    return (pure_unique_count + unique_with_duplicates)/total_rows
