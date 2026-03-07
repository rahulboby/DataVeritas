import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

@st.cache_data(show_spinner=False)
def getOutlierScore(df: pd.DataFrame):
    # Temp Debugger:
    print(" --- Fetching Outlier Score --- ")

    """
    Calculates a single global health score (0 to 1) for the dataset.
    1.0 = No outliers detected.
    0.0 = Every row is an outlier.
    """
    from sections import value_distribution as vdm
    
    # 1. Extract numeric columns
    _, _, _, _, numeric_cols, _, _ = vdm.get_column_stats(df)
    
    if not numeric_cols:
        return 1.0, None # If no numeric data, we assume no outliers
    
    # 2. Prepare data (Impute missing values)
    numeric_df = df[numeric_cols].copy()
    total_rows = len(numeric_df)
    
    if total_rows == 0:
        return 1.0

    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(numeric_df)

    # 3. Fit Isolation Forest
    # contamination='auto' uses a threshold based on the data distribution
    model = IsolationForest(n_estimators=300, contamination='auto', random_state=42)
    model.fit(data_imputed)
    
    # 4. Predict (-1 for outliers, 1 for inliers)
    predictions = model.predict(data_imputed)
    bool_predictions = [True if val == -1 else False for val in predictions]

    # 5. Calculate the score
    # Count how many rows are outliers (-1)
    outlier_count = (predictions == -1).sum()
    
    # Calculate ratio of outliers
    outlier_ratio = outlier_count / total_rows
    
    # Final Score: 1 minus the ratio (Cleanliness score)
    clean_score = 1.0 - outlier_ratio
    outlier_df = df[bool_predictions]
    
    return (float(clean_score), outlier_df)

