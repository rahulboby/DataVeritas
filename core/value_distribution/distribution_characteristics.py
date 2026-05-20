"""
Distribution Characteristics Analysis

Computes advanced statistical metrics for numeric columns:
- Skewness: Distribution shape (-1 to +1, 0=symmetric)
- Kurtosis: Tail heaviness
- Coefficient of Variation: Relative variability (std/mean)
- Variance: Absolute spread
- Median & Percentiles: Distribution summary
- Feature Quality Score: Composite modeling suitability (0-100)
- Entropy & Uniformity: For categorical distributions

These metrics are essential for:
1. Feature selection for machine learning
2. Data transformation decisions
3. Distribution analysis
4. Model readiness assessment
"""

import pandas as pd
import numpy as np


def compute_numeric_distribution_characteristics(df, numeric_columns):
    """
    Compute advanced distribution characteristics for numeric columns.
    
    Args:
        df: Input dataframe
        numeric_columns: List of numeric column names
    
    Returns:
        dict: {column_name: {metric_name: value, ...}, ...}
    """
    characteristics = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        # FIX: Compute all distribution characteristics in one place
        try:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            iqr_outliers = int(((series < lower_fence) | (series > upper_fence)).sum())
            
            mean_val = float(series.mean())
            std_dev = float(series.std())
            variance = float(series.var())
            
            # Skewness: -1 to +1 typical range, 0 = symmetric
            skewness = float(series.skew())
            
            # Kurtosis: excess kurtosis, 0 = normal distribution
            kurtosis = float(series.kurtosis())
            
            # Coefficient of Variation: relative variability (std/mean)
            # Unitless measure for comparing variability across different scales
            cv = (std_dev / abs(mean_val)) if mean_val != 0 else 0
            
            # Unique count for feature quality calculation
            col_unique = df[col].nunique()
            col_null_count = int(df[col].isnull().sum())
            total_rows = len(df)
            
            # Feature Quality Score (0-100) for modeling suitability
            # Factors considered:
            # - Variance: Higher is better (more discriminative)
            # - Missing %: Lower is better (less imputation needed)
            # - Constant columns: Zero information, penalize heavily
            feature_quality_score = 100
            
            if col_null_count > 0:
                # Penalize missing values (up to 50 points)
                missing_pct = (col_null_count / total_rows) * 100
                feature_quality_score -= min(50, int(missing_pct / 2))
            
            if std_dev < 0.001:
                # Penalize near-constant columns (30 points)
                feature_quality_score -= 30
            
            if col_unique <= 1:
                # Zero variance = useless for modeling
                feature_quality_score = 0
            
            feature_quality_score = max(0, feature_quality_score)
            
            characteristics[col] = {
                "mean": round(mean_val, 4),
                "median": round(float(series.median()), 4),
                "std": round(std_dev, 4),
                "variance": round(variance, 4),
                "coefficient_of_variation": round(cv, 4),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "iqr_outlier_count": iqr_outliers,
                "feature_quality_score": feature_quality_score,
                "is_constant": col_unique <= 1,
                "is_near_constant": std_dev < 0.001,
            }
        except Exception as e:
            print(f"Error computing distribution characteristics for {col}: {str(e)}")
            continue
    
    return characteristics


def compute_categorical_distribution_characteristics(df, categorical_columns):
    """
    Compute class balance and distribution uniformity for categorical columns.
    
    Args:
        df: Input dataframe
        categorical_columns: List of categorical column names
    
    Returns:
        dict: {column_name: {metric_name: value, ...}, ...}
    """
    characteristics = {}
    total_rows = len(df)
    
    for col in categorical_columns:
        if col not in df.columns:
            continue
        
        vc = df[col].value_counts(dropna=True)
        
        if len(vc) == 0:
            continue
        
        try:
            # Top values breakdown
            top_5_values = [
                {"value": str(v), "count": int(c), "pct": round(c / total_rows * 100, 2)}
                for v, c in vc.head(5).items()
            ]
            
            # Class Balance Metrics
            # Imbalance Ratio: max_freq / min_freq
            # 1.0 = perfectly balanced, >1 = imbalanced
            if len(vc) > 1:
                imbalance_ratio = float(vc.iloc[0] / vc.iloc[-1])
                # Class Balance Score: inverse of imbalance (0-100)
                # 100 = perfectly balanced, 0 = highly imbalanced
                class_balance_score = round(100 / imbalance_ratio, 1) if imbalance_ratio > 0 else 0
            else:
                imbalance_ratio = 1.0
                class_balance_score = 0.0
            
            # Entropy: Distribution uniformity (higher = more uniform)
            probabilities = vc / vc.sum()
            ent = float(-(probabilities * np.log(probabilities)).sum())
            
            # Distribution Uniformity %: normalized entropy (0-100)
            # 100% = perfectly uniform, 0% = completely concentrated
            if len(vc) > 1:
                max_entropy = np.log(len(vc))
                distribution_uniformity = round((ent / max_entropy) * 100, 1) if max_entropy > 0 else 0
            else:
                distribution_uniformity = 0.0
            
            characteristics[col] = {
                "top_value": str(vc.index[0]),
                "top_frequency": int(vc.iloc[0]),
                "top_share_pct": round(vc.iloc[0] / total_rows * 100, 2) if total_rows > 0 else 0,
                "top_5_values": top_5_values,
                "imbalance_ratio": round(imbalance_ratio, 2),
                "class_balance_score": class_balance_score,
                "entropy": round(ent, 4),
                "distribution_uniformity": distribution_uniformity,
                "num_classes": len(vc),
            }
        except Exception as e:
            print(f"Error computing categorical characteristics for {col}: {str(e)}")
            continue
    
    return characteristics
