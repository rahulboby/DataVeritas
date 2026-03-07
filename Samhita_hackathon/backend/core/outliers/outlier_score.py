import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from functools import lru_cache


def get_column_stats(df):
    """Categorizes columns by type. Standalone version without Streamlit dependency."""
    all_cols = df.columns.tolist()

    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    remaining = [col for col in all_cols if col not in constant_cols]

    datetime_cols = df[remaining].select_dtypes(include=['datetime', 'datetimetz']).columns.tolist() if remaining else []
    remaining = [col for col in remaining if col not in datetime_cols]

    boolean_cols = df[remaining].select_dtypes(include=['bool']).columns.tolist() if remaining else []
    remaining = [col for col in remaining if col not in boolean_cols]

    numeric_cols = df[remaining].select_dtypes(include=['number']).columns.tolist() if remaining else []
    remaining = [col for col in remaining if col not in numeric_cols]

    categorical_cols = [col for col in remaining if df[col].nunique() <= (df.shape[0] / 4)]
    remaining = [col for col in remaining if col not in categorical_cols]

    identifier_cols = list(remaining)

    counts = {
        "Numeric": len(numeric_cols),
        "Categorical": len(categorical_cols),
        "Datetime": len(datetime_cols),
        "Boolean": len(boolean_cols),
        "Constant": len(constant_cols),
        "Identifier": len(identifier_cols)
    }

    return counts, constant_cols, datetime_cols, boolean_cols, numeric_cols, categorical_cols, identifier_cols


def get_outlier_score(df: pd.DataFrame):
    """
    Calculates a global health score (0 to 1) for the dataset based on outlier detection.
    
    Uses Isolation Forest on numeric columns.
    1.0 = No outliers detected.
    0.0 = Every row is an outlier.
    
    Returns:
        Tuple of (clean_score: float, outlier_df: DataFrame or None)
    """
    # Extract numeric columns
    _, _, _, _, numeric_cols, _, _ = get_column_stats(df)

    if not numeric_cols:
        return 1.0, None

    numeric_df = df[numeric_cols].copy()
    total_rows = len(numeric_df)

    if total_rows == 0:
        return 1.0, None

    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    data_imputed = imputer.fit_transform(numeric_df)

    # Fit Isolation Forest
    model = IsolationForest(n_estimators=300, contamination='auto', random_state=42)
    model.fit(data_imputed)

    # Predict (-1 for outliers, 1 for inliers)
    predictions = model.predict(data_imputed)
    bool_predictions = [True if val == -1 else False for val in predictions]

    # Calculate score
    outlier_count = (predictions == -1).sum()
    outlier_ratio = outlier_count / total_rows
    clean_score = 1.0 - outlier_ratio

    outlier_df = df[bool_predictions]

    return (float(clean_score), outlier_df)
