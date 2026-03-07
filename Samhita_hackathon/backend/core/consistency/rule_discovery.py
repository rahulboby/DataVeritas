import pandas as pd
import numpy as np


def discover_rules(df):
    """
    Automatically discover data quality rules from the dataset.
    
    Analyzes:
    - Date column ordering relationships
    - Numeric column range constraints  
    - Conditional patterns (IF X THEN Y)
    - Uniqueness candidates
    - Correlation-based relationships
    
    Returns:
        List of discovered rule dicts compatible with the rule engine.
    """
    discovered_rules = []
    
    # 1. Discover date ordering rules
    discovered_rules.extend(_discover_date_ordering(df))
    
    # 2. Discover uniqueness candidates
    discovered_rules.extend(_discover_uniqueness_candidates(df))
    
    # 3. Discover numeric range constraints
    discovered_rules.extend(_discover_range_constraints(df))
    
    # 4. Discover conditional rules (IF-THEN patterns)
    discovered_rules.extend(_discover_conditional_rules(df))
    
    # 5. Discover cross-field comparisons (numeric)
    discovered_rules.extend(_discover_numeric_relationships(df))
    
    return discovered_rules


def _discover_date_ordering(df):
    """Find date columns that follow a consistent temporal ordering."""
    rules = []
    date_cols = []
    
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notnull().sum() > len(df) * 0.8:  # At least 80% parseable
                date_cols.append(col)
        except Exception:
            continue
    
    # Check pairs for consistent ordering
    for i in range(len(date_cols)):
        for j in range(i + 1, len(date_cols)):
            col1, col2 = date_cols[i], date_cols[j]
            d1 = pd.to_datetime(df[col1], errors='coerce')
            d2 = pd.to_datetime(df[col2], errors='coerce')
            
            valid = d1.notnull() & d2.notnull()
            if valid.sum() < 10:
                continue
            
            lt_ratio = (d1[valid] < d2[valid]).mean()
            gt_ratio = (d1[valid] > d2[valid]).mean()
            
            if lt_ratio > 0.90:
                rules.append({
                    "name": f"Auto: {col1} < {col2}",
                    "rule_type": "Cross-Field Comparison",
                    "field": col1,
                    "field2": col2,
                    "operator": "<",
                    "enabled": True,
                    "confidence": round(float(lt_ratio), 3),
                    "auto_discovered": True,
                })
            elif gt_ratio > 0.90:
                rules.append({
                    "name": f"Auto: {col1} > {col2}",
                    "rule_type": "Cross-Field Comparison",
                    "field": col1,
                    "field2": col2,
                    "operator": ">",
                    "enabled": True,
                    "confidence": round(float(gt_ratio), 3),
                    "auto_discovered": True,
                })
    
    return rules


def _discover_uniqueness_candidates(df):
    """Find columns that are likely primary keys (very high cardinality, no nulls)."""
    rules = []
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            continue
        
        cardinality_ratio = df[col].nunique() / len(df)
        if cardinality_ratio > 0.98:  # Near-unique
            rules.append({
                "name": f"Auto: {col} should be unique",
                "rule_type": "Uniqueness",
                "field": col,
                "enabled": True,
                "confidence": round(float(cardinality_ratio), 3),
                "auto_discovered": True,
            })
    
    return rules


def _discover_range_constraints(df):
    """Infer range constraints for numeric columns using IQR."""
    rules = []
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 20:
            continue
        
        q1 = values.quantile(0.01)
        q99 = values.quantile(0.99)
        
        # Only add if there's a meaningful range
        if q1 != q99:
            rules.append({
                "name": f"Auto: {col} range [{q1:.2f}, {q99:.2f}]",
                "rule_type": "Range",
                "field": col,
                "min_value": round(float(q1), 2),
                "max_value": round(float(q99), 2),
                "enabled": True,
                "confidence": 0.95,
                "auto_discovered": True,
            })
    
    return rules


def _discover_conditional_rules(df):
    """Discover IF-THEN patterns in categorical columns."""
    rules = []
    
    cat_cols = [col for col in df.columns 
                if df[col].dtype == 'object' and df[col].nunique() <= 20]
    
    for i in range(len(cat_cols)):
        for j in range(len(cat_cols)):
            if i == j:
                continue
            
            col_if = cat_cols[i]
            col_then = cat_cols[j]
            
            # For each value of col_if, check if col_then is always the same
            for val_if in df[col_if].dropna().unique():
                subset = df[df[col_if] == val_if][col_then].dropna()
                if len(subset) < 10:
                    continue
                
                mode_val = subset.mode()
                if len(mode_val) == 0:
                    continue
                
                mode_ratio = (subset == mode_val.iloc[0]).mean()
                
                if mode_ratio > 0.95:
                    rules.append({
                        "name": f"Auto: IF {col_if}='{val_if}' THEN {col_then}='{mode_val.iloc[0]}'",
                        "rule_type": "Conditional (IF-THEN)",
                        "field": col_if,
                        "condition_value": str(val_if),
                        "then_field": col_then,
                        "then_value": str(mode_val.iloc[0]),
                        "enabled": True,
                        "confidence": round(float(mode_ratio), 3),
                        "auto_discovered": True,
                    })
    
    return rules


def _discover_numeric_relationships(df):
    """Discover relationships between numeric columns using correlation."""
    rules = []
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return rules
    
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col1 = numeric_cols[i]
            col2 = numeric_cols[j]
            
            valid = df[[col1, col2]].dropna()
            if len(valid) < 20:
                continue
            
            # Check if one is consistently >= the other
            ge_ratio = (valid[col1] >= valid[col2]).mean()
            le_ratio = (valid[col1] <= valid[col2]).mean()
            
            if ge_ratio > 0.95:
                rules.append({
                    "name": f"Auto: {col1} >= {col2}",
                    "rule_type": "Cross-Field Comparison",
                    "field": col1,
                    "field2": col2,
                    "operator": ">=",
                    "enabled": True,
                    "confidence": round(float(ge_ratio), 3),
                    "auto_discovered": True,
                })
            elif le_ratio > 0.95:
                rules.append({
                    "name": f"Auto: {col1} <= {col2}",
                    "rule_type": "Cross-Field Comparison",
                    "field": col1,
                    "field2": col2,
                    "operator": "<=",
                    "enabled": True,
                    "confidence": round(float(le_ratio), 3),
                    "auto_discovered": True,
                })
    
    return rules
