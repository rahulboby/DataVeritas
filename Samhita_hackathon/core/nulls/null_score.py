
def getNullScore(df):
    # Temp Debugger:
    print(" --- Fetching Null Score --- ")
    
    return len(df.dropna())/df.shape[0]