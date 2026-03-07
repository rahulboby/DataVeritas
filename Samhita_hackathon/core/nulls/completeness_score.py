def getCompletenessScore(df):
    # Temp Debugger:
    print(" --- Fetching Completeness Score --- ")
    
    total_nulls = df.isnull().sum().sum()
    nrows = df.shape[0]
    ncols = df.shape[1]
    total_cells = nrows * ncols
    return(float((total_cells - total_nulls) / total_cells)) if total_cells > 0 else 0

