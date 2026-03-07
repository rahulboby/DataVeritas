import pandas as pd
import numpy as np
from datetime import datetime
import re
import streamlit as st


def validate_mandatory_field(df, field_name):
    """Check if field has no null values."""
    violations = df[df[field_name].isnull()].copy()
    violations['violation_reason'] = f"{field_name}: Missing value (null)"
    violations['violated_field'] = field_name
    return violations


def validate_datatype(df, field_name, expected_type):
    """Check if field matches expected data type."""
    violations = pd.DataFrame()
    
    type_map = {
        "String": str,
        "Integer": (int, np.integer),
        "Float": (float, np.floating),
        "Date": 'datetime',
        "Boolean": bool
    }
    
    if expected_type == "Date":
        # Try to convert to datetime, failures are violations
        temp = pd.to_datetime(df[field_name], errors='coerce')
        mask = temp.isnull() & df[field_name].notnull()
        violations = df[mask].copy()
        violations['violation_reason'] = f"{field_name}: Invalid date format"
    else:
        expected = type_map.get(expected_type, str)
        mask = ~df[field_name].apply(lambda x: isinstance(x, expected) if pd.notnull(x) else True)
        violations = df[mask].copy()
        violations['violation_reason'] = f"{field_name}: Expected {expected_type}"
    
    violations['violated_field'] = field_name
    return violations


def validate_format(df, field_name, format_type):
    """Check if field matches expected format pattern."""
    violations = pd.DataFrame()
    
    patterns = {
        "Email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "Phone": r'^\+?1?\d{9,15}$',
        "URL": r'^https?://[^\s]+$',
        "Zipcode": r'^\d{5}(-\d{4})?$',
        "Custom": None  # User can define
    }
    
    pattern = patterns.get(format_type, format_type)
    
    if pattern:
        mask = ~df[field_name].astype(str).str.match(pattern, na=False)
        violations = df[mask & df[field_name].notnull()].copy()
        violations['violation_reason'] = f"{field_name}: Invalid {format_type} format"
        violations['violated_field'] = field_name
    
    return violations


def validate_range(df, field_name, min_val, max_val):
    """Check if numeric field is within range."""
    violations = pd.DataFrame()
    
    try:
        if min_val is not None:
            mask_min = df[field_name] < min_val
            violations = pd.concat([violations, df[mask_min].copy()])
        
        if max_val is not None:
            mask_max = df[field_name] > max_val
            violations = pd.concat([violations, df[mask_max].copy()])
        
        violations['violation_reason'] = f"{field_name}: Out of range [{min_val}, {max_val}]"
        violations['violated_field'] = field_name
    except:
        pass
    
    return violations.drop_duplicates()


def validate_uniqueness(df, field_name):
    """Check if field has unique values."""
    duplicates = df[df.duplicated(subset=[field_name], keep=False)].copy()
    duplicates['violation_reason'] = f"{field_name}: Duplicate value found"
    duplicates['violated_field'] = field_name
    return duplicates


def validate_cross_field(df, field1, operator, field2):
    """Check relationships between two fields."""
    violations = pd.DataFrame()
    
    try:
        if operator == "<":
            mask = df[field1] >= df[field2]
        elif operator == ">":
            mask = df[field1] <= df[field2]
        elif operator == "<=":
            mask = df[field1] > df[field2]
        elif operator == ">=":
            mask = df[field1] < df[field2]
        elif operator == "==":
            mask = df[field1] != df[field2]
        elif operator == "!=":
            mask = df[field1] == df[field2]
        else:
            mask = pd.Series([False] * len(df))
        
        violations = df[mask].copy()
        violations['violation_reason'] = f"{field1} must be {operator} {field2}"
        violations['violated_field'] = f"{field1}, {field2}"
    except:
        pass
    
    return violations


def validate_conditional(df, condition_field, condition_value, then_field, then_value):
    """Check IF-THEN conditional rules."""
    violations = pd.DataFrame()
    
    try:
        # Find rows where condition is met
        condition_mask = df[condition_field] == condition_value
        # Among those, find violations of the THEN clause
        then_mask = df[then_field] != then_value
        
        violations = df[condition_mask & then_mask].copy()
        violations['violation_reason'] = f"IF {condition_field}='{condition_value}' THEN {then_field} must be '{then_value}'"
        violations['violated_field'] = f"{condition_field}, {then_field}"
    except:
        pass
    
    return violations


def apply_rules(df, rules):
    """
    Apply all enabled rules and return violations.
    
    Args:
        df: Input dataframe
        rules: List of rule dictionaries from session state
    
    Returns:
        all_violations: DataFrame with all violations
        violation_summary: Dict with per-rule violation counts
    """
    all_violations = pd.DataFrame()
    violation_summary = {}
    
    for rule in rules:
        if not rule.get('enabled', True):
            continue
        
        rule_name = rule['name']
        rule_type = rule['rule_type']
        field = rule.get('field')
        
        violations = pd.DataFrame()
        
        try:
            if rule_type == "Mandatory Field":
                violations = validate_mandatory_field(df, field)
            
            elif rule_type == "Data Type":
                expected_type = rule.get('expected_type', 'String')
                violations = validate_datatype(df, field, expected_type)
            
            elif rule_type == "Format":
                format_type = rule.get('format_type', 'Email')
                violations = validate_format(df, field, format_type)
            
            elif rule_type == "Range":
                min_val = rule.get('min_value')
                max_val = rule.get('max_value')
                violations = validate_range(df, field, min_val, max_val)
            
            elif rule_type == "Uniqueness":
                violations = validate_uniqueness(df, field)
            
            elif rule_type == "Cross-Field Comparison":
                field2 = rule.get('field2')
                operator = rule.get('operator', '<')
                violations = validate_cross_field(df, field, operator, field2)
            
            elif rule_type == "Conditional (IF-THEN)":
                condition_value = rule.get('condition_value')
                then_field = rule.get('then_field')
                then_value = rule.get('then_value')
                violations = validate_conditional(df, field, condition_value, then_field, then_value)
            
            if not violations.empty:
                violations['rule_name'] = rule_name
                all_violations = pd.concat([all_violations, violations], ignore_index=True)
                violation_summary[rule_name] = len(violations)
            else:
                violation_summary[rule_name] = 0
        
        except Exception as e:
            st.error(f"Error applying rule '{rule_name}': {str(e)}")
    
    return all_violations, violation_summary
