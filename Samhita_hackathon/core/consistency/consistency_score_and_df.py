import pandas as pd
import streamlit as st
from core.consistency.rule_engine import apply_rules


def getConsistencyScore(df):
    """
    Calculate consistency score based on configured rules.
    
    Returns:
        violation_score: Float between 0-1 (1.0 = perfect, 0.0 = all violations)
        violation_df: DataFrame containing rows with violations
    """
    
    # Check if rules exist in session state
    if 'consistency_rules' not in st.session_state or not st.session_state.consistency_rules:
        # No rules configured - use default hardcoded rules
        return getConsistencyScore_Default(df)
    
    # Apply dynamic rules
    try:
        violation_df, violation_summary = apply_rules(
            df, 
            st.session_state.consistency_rules
        )
        
        if violation_df.empty:
            # No violations found
            return 1.0, pd.DataFrame()
        
        # Get unique violated rows (since one row can violate multiple rules)
        violated_indices = violation_df.index.unique()
        violations_count = len(violated_indices)
        total_rows = len(df)
        
        # Calculate score (1.0 = perfect, 0.0 = all bad)
        violation_score = 1 - (violations_count / total_rows) if total_rows > 0 else 1.0
        
        # Return unique violated rows
        unique_violations = df.loc[violated_indices].copy()
        
        # Add violation metadata if needed
        violation_details = violation_df.groupby(violation_df.index).agg({
            'violation_reason': lambda x: ' | '.join(x),
            'violated_field': lambda x: ', '.join(set(str(v) for v in x)),
            'rule_name': lambda x: ', '.join(set(x))
        }).reset_index()
        
        unique_violations = unique_violations.merge(
            violation_details,
            left_index=True,
            right_on='index',
            how='left'
        ).drop('index', axis=1)
        
        return violation_score, unique_violations
    
    except Exception as e:
        st.error(f"Error calculating consistency score: {str(e)}")
        return 1.0, pd.DataFrame()


def getConsistencyScore_Default(df):
    """
    Fallback function with hardcoded business rules.
    Used when no dynamic rules are configured.
    
    Original rules:
    - order_date < delivery_date < last_service_date
    - IF fuel_type == "hybrid" THEN transmission == "automatic"
    - order_number is unique
    - Same customer_id should have same email, phone, customer_name
    """
    
    all_violations = pd.DataFrame()
    
    # Rule 1: Date sequence validation (if these columns exist)
    if all(col in df.columns for col in ['order_date', 'delivery_date', 'last_service_date']):
        try:
            # Convert to datetime
            order_dates = pd.to_datetime(df['order_date'], errors='coerce')
            delivery_dates = pd.to_datetime(df['delivery_date'], errors='coerce')
            service_dates = pd.to_datetime(df['last_service_date'], errors='coerce')
            
            # Check violations
            mask = (order_dates >= delivery_dates) | (delivery_dates >= service_dates)
            violations = df[mask].copy()
            violations['violation_reason'] = "Date sequence: order_date < delivery_date < last_service_date violated"
            violations['violated_field'] = "order_date, delivery_date, last_service_date"
            
            all_violations = pd.concat([all_violations, violations], ignore_index=True)
        except:
            pass
    
    # Rule 2: Hybrid vehicles must be automatic
    if all(col in df.columns for col in ['fuel_type', 'transmission']):
        try:
            mask = (df['fuel_type'].str.lower() == 'hybrid') & (df['transmission'].str.lower() != 'automatic')
            violations = df[mask].copy()
            violations['violation_reason'] = "Hybrid vehicles must have automatic transmission"
            violations['violated_field'] = "fuel_type, transmission"
            
            all_violations = pd.concat([all_violations, violations], ignore_index=True)
        except:
            pass
    
    # Rule 3: Order number uniqueness
    if 'order_number' in df.columns:
        try:
            duplicates = df[df.duplicated(subset=['order_number'], keep=False)].copy()
            duplicates['violation_reason'] = "Duplicate order_number found"
            duplicates['violated_field'] = "order_number"
            
            all_violations = pd.concat([all_violations, duplicates], ignore_index=True)
        except:
            pass
    
    # Rule 4: Customer consistency (same customer_id should have same details)
    if all(col in df.columns for col in ['customer_id', 'email', 'customer_name']):
        try:
            # Group by customer_id and check for inconsistencies
            customer_groups = df.groupby('customer_id').agg({
                'email': lambda x: x.nunique(),
                'customer_name': lambda x: x.nunique()
            })
            
            # Find customer_ids with multiple values
            inconsistent_customers = customer_groups[
                (customer_groups['email'] > 1) | 
                (customer_groups['customer_name'] > 1)
            ].index
            
            violations = df[df['customer_id'].isin(inconsistent_customers)].copy()
            violations['violation_reason'] = "Same customer_id has different email/name"
            violations['violated_field'] = "customer_id, email, customer_name"
            
            all_violations = pd.concat([all_violations, violations], ignore_index=True)
        except:
            pass
    
    # Calculate score
    if all_violations.empty:
        return 1.0, pd.DataFrame()
    
    # Get unique violated rows
    violated_indices = all_violations.index.unique()
    violations_count = len(violated_indices)
    total_rows = len(df)
    
    violation_score = 1 - (violations_count / total_rows) if total_rows > 0 else 1.0
    
    # Return unique violations with aggregated reasons
    unique_violations = df.loc[violated_indices].copy()
    
    violation_details = all_violations.groupby(all_violations.index).agg({
        'violation_reason': lambda x: ' | '.join(x),
        'violated_field': lambda x: ', '.join(set(str(v) for v in x))
    }).reset_index()
    
    unique_violations = unique_violations.merge(
        violation_details,
        left_index=True,
        right_on='index',
        how='left'
    ).drop('index', axis=1)
    
    return violation_score, unique_violations

