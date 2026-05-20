"""
Correlation & Feature Relationship Analysis

Computes correlation matrices and identifies feature relationships:
- Pearson Correlation: Linear relationships between numeric features
- Top Correlations: Highest correlated features per column
- Multicollinearity Detection: Identify redundant features

Used for:
1. Feature selection (avoiding redundant features)
2. Understanding feature relationships
3. Detecting multicollinearity
4. Guiding feature engineering decisions
"""

import pandas as pd
import numpy as np


def compute_correlation_matrix(df, numeric_columns):
    """
    Compute Pearson correlation matrix for numeric columns.
    
    Args:
        df: Input dataframe
        numeric_columns: List of numeric column names
    
    Returns:
        dict: {column: {correlated_col: correlation_value, ...}, ...}
               Top 5 correlated features per column (excluding self)
    """
    correlation_data = {}
    
    if not numeric_columns or len(numeric_columns) < 2:
        return correlation_data
    
    try:
        # Select only numeric columns that exist in dataframe
        numeric_df = df[numeric_columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df) < 2:
            return correlation_data
        
        # Compute Pearson correlation matrix
        pearson_corr = numeric_df.corr(method='pearson')
        
        # For each column, extract top correlated features
        for col in numeric_df.columns:
            # Get correlations with this column, drop self-correlation
            corr_with_col = pearson_corr[col].drop(col).abs().sort_values(ascending=False)
            
            # Get top 5 most correlated features
            if len(corr_with_col) > 0:
                top_correlations = corr_with_col.head(5).to_dict()
                correlation_data[col] = top_correlations
    
    except Exception as e:
        print(f"Error computing correlation matrix: {str(e)}")
    
    return correlation_data


def identify_high_correlations(df, numeric_columns, threshold=0.8):
    """
    Identify pairs of features with high correlation (potential multicollinearity).
    
    Args:
        df: Input dataframe
        numeric_columns: List of numeric column names
        threshold: Correlation threshold (default 0.8)
    
    Returns:
        list: [{col1, col2, correlation}, ...]
    """
    high_correlations = []
    
    if not numeric_columns or len(numeric_columns) < 2:
        return high_correlations
    
    try:
        numeric_df = df[numeric_columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty or len(numeric_df) < 2:
            return high_correlations
        
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Find pairs with correlation > threshold (excluding diagonal)
        for i in range(len(pearson_corr.columns)):
            for j in range(i + 1, len(pearson_corr.columns)):
                col1 = pearson_corr.columns[i]
                col2 = pearson_corr.columns[j]
                corr_value = abs(pearson_corr.iloc[i, j])
                
                if corr_value >= threshold:
                    high_correlations.append({
                        "column_1": col1,
                        "column_2": col2,
                        "correlation": round(corr_value, 4),
                    })
        
        # Sort by correlation strength (descending)
        high_correlations.sort(key=lambda x: x["correlation"], reverse=True)
    
    except Exception as e:
        print(f"Error identifying high correlations: {str(e)}")
    
    return high_correlations
