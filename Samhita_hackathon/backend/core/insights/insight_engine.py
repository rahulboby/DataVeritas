import pandas as pd
import numpy as np


def generate_insights(df, rules=None):
    """
    Generate human-readable data quality insights from the dataset.
    
    Aggregates results from all analytics dimensions and produces
    ranked, actionable insight strings.
    
    Returns:
        List of insight dicts: [{severity, category, message, value}, ...]
    """
    insights = []
    
    # 1. Null / Completeness insights
    insights.extend(_null_insights(df))
    
    # 2. Duplicate insights
    insights.extend(_duplicate_insights(df))
    
    # 3. Outlier insights
    insights.extend(_outlier_insights(df))
    
    # 4. Consistency insights (if rules provided)
    if rules:
        insights.extend(_consistency_insights(df, rules))
    
    # 5. Distribution insights
    insights.extend(_distribution_insights(df))
    
    # 6. Schema insights
    insights.extend(_schema_insights(df))
    
    # Sort by severity (critical first)
    severity_order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    insights.sort(key=lambda x: severity_order.get(x["severity"], 4))
    
    return insights


def _null_insights(df):
    insights = []
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = int(df.isnull().sum().sum())
    null_pct = total_nulls / total_cells * 100 if total_cells > 0 else 0
    
    if null_pct > 20:
        insights.append({
            "severity": "critical",
            "category": "Completeness",
            "message": f"{null_pct:.1f}% of all data cells are missing — dataset has severe completeness issues",
            "value": round(null_pct, 1)
        })
    elif null_pct > 5:
        insights.append({
            "severity": "warning",
            "category": "Completeness",
            "message": f"{null_pct:.1f}% of data cells are missing",
            "value": round(null_pct, 1)
        })
    elif null_pct > 0:
        insights.append({
            "severity": "info",
            "category": "Completeness",
            "message": f"Only {null_pct:.1f}% missing data — good completeness",
            "value": round(null_pct, 1)
        })
    else:
        insights.append({
            "severity": "good",
            "category": "Completeness",
            "message": "Dataset is 100% complete — no missing values",
            "value": 0
        })
    
    # Worst columns
    null_per_col = df.isnull().sum()
    worst_cols = null_per_col[null_per_col > 0].sort_values(ascending=False)
    for col in worst_cols.head(3).index:
        col_null_pct = null_per_col[col] / len(df) * 100
        if col_null_pct > 10:
            insights.append({
                "severity": "warning",
                "category": "Completeness",
                "message": f"Column '{col}' has {col_null_pct:.1f}% missing values",
                "value": round(col_null_pct, 1)
            })
    
    return insights


def _duplicate_insights(df):
    insights = []
    dup_count = int(df.duplicated().sum())
    dup_pct = dup_count / len(df) * 100 if len(df) > 0 else 0
    
    if dup_pct > 10:
        insights.append({
            "severity": "critical",
            "category": "Uniqueness",
            "message": f"{dup_count:,} duplicate rows detected ({dup_pct:.1f}% of dataset)",
            "value": dup_count
        })
    elif dup_count > 0:
        insights.append({
            "severity": "warning",
            "category": "Uniqueness",
            "message": f"{dup_count:,} duplicate rows found ({dup_pct:.1f}%)",
            "value": dup_count
        })
    else:
        insights.append({
            "severity": "good",
            "category": "Uniqueness",
            "message": "No exact duplicate rows found",
            "value": 0
        })
    
    return insights


def _outlier_insights(df):
    insights = []
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        insights.append({
            "severity": "info",
            "category": "Outliers",
            "message": "No numeric columns — outlier analysis not applicable",
            "value": 0
        })
        return insights
    
    total_outliers = 0
    outlier_cols = []
    
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 10:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = int(((values < lower) | (values > upper)).sum())
        if n_outliers > 0:
            total_outliers += n_outliers
            outlier_cols.append((col, n_outliers))
    
    if total_outliers > 0:
        insights.append({
            "severity": "warning" if total_outliers / len(df) < 0.1 else "critical",
            "category": "Outliers",
            "message": f"{total_outliers:,} outlier values detected across {len(outlier_cols)} columns",
            "value": total_outliers
        })
        
        # Top outlier columns
        outlier_cols.sort(key=lambda x: x[1], reverse=True)
        for col, count in outlier_cols[:3]:
            insights.append({
                "severity": "info",
                "category": "Outliers",
                "message": f"Column '{col}' has {count:,} outlier values",
                "value": count
            })
    else:
        insights.append({
            "severity": "good",
            "category": "Outliers",
            "message": "No significant outliers detected in numeric columns",
            "value": 0
        })
    
    return insights


def _consistency_insights(df, rules):
    insights = []
    from core.consistency.rule_engine import apply_rules
    
    try:
        violations_df, summary = apply_rules(df, rules)
        total_violations = len(violations_df)
        
        if total_violations > 0:
            insights.append({
                "severity": "warning",
                "category": "Consistency",
                "message": f"{total_violations:,} rule violations detected across {len(summary)} rules",
                "value": total_violations
            })
            
            for rule_name, count in summary.items():
                if count > 0:
                    pct = count / len(df) * 100
                    insights.append({
                        "severity": "info",
                        "category": "Consistency",
                        "message": f"Rule '{rule_name}': {count:,} violations ({pct:.1f}%)",
                        "value": count
                    })
        else:
            insights.append({
                "severity": "good",
                "category": "Consistency",
                "message": "All consistency rules pass — no violations",
                "value": 0
            })
    except Exception:
        pass
    
    return insights


def _distribution_insights(df):
    insights = []
    
    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            insights.append({
                "severity": "warning",
                "category": "Distribution",
                "message": f"Column '{col}' has only 1 unique value (constant) — may be useless",
                "value": 1
            })
    
    # Check for high cardinality text columns
    for col in df.select_dtypes(include=['object']).columns:
        ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        if ratio > 0.95:
            insights.append({
                "severity": "info",
                "category": "Distribution",
                "message": f"Column '{col}' has very high cardinality ({ratio:.0%}) — likely an identifier",
                "value": round(ratio, 3)
            })
    
    return insights


def _schema_insights(df):
    insights = []
    
    insights.append({
        "severity": "info",
        "category": "Schema",
        "message": f"Dataset has {len(df):,} rows × {len(df.columns)} columns",
        "value": len(df)
    })
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    if memory_mb > 100:
        insights.append({
            "severity": "warning",
            "category": "Schema",
            "message": f"Dataset uses {memory_mb:.1f} MB of memory — consider optimization",
            "value": round(memory_mb, 1)
        })
    
    return insights
