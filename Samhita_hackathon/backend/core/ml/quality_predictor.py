import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def extract_meta_features(df):
    """
    Extract dataset-level meta-features for quality prediction.
    
    Returns dict with 7 features.
    """
    total_cells = df.shape[0] * df.shape[1]
    numeric_cols = df.select_dtypes(include=['number']).columns

    features = {
        "null_ratio": float(df.isnull().sum().sum() / total_cells) if total_cells > 0 else 0,
        "duplicate_ratio": float(df.duplicated().sum() / df.shape[0]) if df.shape[0] > 0 else 0,
        "outlier_ratio": _estimate_outlier_ratio(df, numeric_cols),
        "mean_entropy": _mean_column_entropy(df),
        "avg_cardinality_ratio": _avg_cardinality_ratio(df),
        "skewness_mean": float(df[numeric_cols].skew().abs().mean()) if len(numeric_cols) > 0 else 0,
        "kurtosis_mean": float(df[numeric_cols].kurtosis().abs().mean()) if len(numeric_cols) > 0 else 0,
    }
    return features


def predict_quality(df):
    """
    Predict dataset quality score using meta-features and a trained model.
    
    Since we don't have labeled training data, we generate synthetic datasets
    with known quality issues and train on them.
    
    Returns:
        dict with predicted_quality_score and feature_importance
    """
    # Extract features from input dataset
    input_features = extract_meta_features(df)
    
    # Generate training data (synthetic datasets with known quality levels)
    X_train, y_train = _generate_training_data(n_samples=200)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    
    # Predict
    feature_names = list(input_features.keys())
    X_input = np.array([[input_features[f] for f in feature_names]])
    predicted_score = float(np.clip(model.predict(X_input)[0], 0, 1))
    
    # Feature importance
    importances = dict(zip(feature_names, [round(float(v), 4) for v in model.feature_importances_]))
    
    return {
        "predicted_quality_score": round(predicted_score, 4),
        "feature_importance": importances,
        "input_features": {k: round(v, 4) for k, v in input_features.items()},
    }


def _estimate_outlier_ratio(df, numeric_cols):
    """Quick IQR-based outlier ratio estimate (faster than Isolation Forest)."""
    if len(numeric_cols) == 0:
        return 0.0
    
    total_outliers = 0
    total_values = 0
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        total_outliers += int(((values < lower) | (values > upper)).sum())
        total_values += len(values)
    
    return total_outliers / total_values if total_values > 0 else 0.0


def _mean_column_entropy(df):
    """Calculate mean Shannon entropy across all columns."""
    from scipy.stats import entropy
    
    entropies = []
    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        if len(value_counts) > 1:
            entropies.append(float(entropy(value_counts.values)))
    
    return float(np.mean(entropies)) if entropies else 0.0


def _avg_cardinality_ratio(df):
    """Average cardinality ratio across all columns."""
    if df.shape[0] == 0:
        return 0.0
    ratios = [df[col].nunique() / df.shape[0] for col in df.columns]
    return float(np.mean(ratios))


def _generate_training_data(n_samples=200):
    """
    Generate synthetic training data for the quality predictor.
    Each sample has 7 meta-features and a known quality score.
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        null_ratio = np.random.beta(2, 10)  # mostly low
        dup_ratio = np.random.beta(2, 10)
        outlier_ratio = np.random.beta(2, 15)
        mean_entropy = np.random.uniform(0, 5)
        cardinality = np.random.uniform(0, 1)
        skewness = np.random.exponential(1)
        kurtosis = np.random.exponential(2)
        
        # Quality score is inversely related to problems
        quality = (
            (1 - null_ratio) * 0.25
            + (1 - dup_ratio) * 0.25
            + (1 - outlier_ratio) * 0.15
            + min(mean_entropy / 3, 1) * 0.15  # moderate entropy is good
            + cardinality * 0.1
            + max(0, 1 - skewness / 5) * 0.05
            + max(0, 1 - kurtosis / 10) * 0.05
        )
        quality = np.clip(quality + np.random.normal(0, 0.03), 0, 1)
        
        X.append([null_ratio, dup_ratio, outlier_ratio, mean_entropy, cardinality, skewness, kurtosis])
        y.append(quality)
    
    return np.array(X), np.array(y)
