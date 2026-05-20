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
        
        # FIX: Get unique violated rows using index
        # The violation_df may have multiple rows for the same original row (one per violated rule)
        # We need to deduplicate by original dataframe index
        violated_indices = violation_df.index.unique()
        violations_count = len(violated_indices)
        total_rows = len(df)
        
        # Calculate score (1.0 = perfect, 0.0 = all bad)
        violation_score = 1 - (violations_count / total_rows) if total_rows > 0 else 1.0
        
        # Return unique violated rows with aggregated violation metadata
        unique_violations = df.loc[violated_indices].copy()
        
        # FIX: Properly aggregate violation details by index
        # Group violations by original row index and aggregate the metadata
        violation_details = violation_df.groupby(violation_df.index).agg({
            'violation_reason': lambda x: ' | '.join(x),
            'violated_field': lambda x: ', '.join(set(str(v) for v in x if v)),
            'rule_name': lambda x: ', '.join(set(x))
        }).reset_index()
        
        # FIX: Use index-based merge instead of column-based merge
        # This prevents index misalignment issues
        unique_violations = unique_violations.merge(
            violation_details.rename(columns={'index': 'original_index'}),
            left_index=True,
            right_on='original_index',
            how='left'
        ).drop('original_index', axis=1)
        
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
    
    # FIX: Use a dictionary to track ALL violations per row index
    # This preserves the original dataframe indices and prevents index loss
    all_violations_by_index = {}  # {row_index: {rule_name, violation_reason, violated_field}}
    
    # Rule 1: Date sequence validation (if these columns exist)
    if all(col in df.columns for col in ['order_date', 'delivery_date', 'last_service_date']):
        try:
            # Convert to datetime
            order_dates = pd.to_datetime(df['order_date'], errors='coerce')
            delivery_dates = pd.to_datetime(df['delivery_date'], errors='coerce')
            service_dates = pd.to_datetime(df['last_service_date'], errors='coerce')
            
            # Check violations: order_date < delivery_date < last_service_date
            mask = (order_dates >= delivery_dates) | (delivery_dates >= service_dates)
            violated_indices = df[mask].index
            
            for idx in violated_indices:
                if idx not in all_violations_by_index:
                    all_violations_by_index[idx] = []
                all_violations_by_index[idx].append({
                    'rule_name': 'Date Sequence',
                    'violation_reason': "Date sequence: order_date < delivery_date < last_service_date violated",
                    'violated_field': "order_date, delivery_date, last_service_date"
                })
        except Exception as e:
            st.warning(f"Error checking date sequence rule: {str(e)}")
    
    # Rule 2: Hybrid vehicles must be automatic
    if all(col in df.columns for col in ['fuel_type', 'transmission']):
        try:
            mask = (df['fuel_type'].str.lower() == 'hybrid') & (df['transmission'].str.lower() != 'automatic')
            violated_indices = df[mask].index
            
            for idx in violated_indices:
                if idx not in all_violations_by_index:
                    all_violations_by_index[idx] = []
                all_violations_by_index[idx].append({
                    'rule_name': 'Fuel Type Constraint',
                    'violation_reason': "Hybrid vehicles must have automatic transmission",
                    'violated_field': "fuel_type, transmission"
                })
        except Exception as e:
            st.warning(f"Error checking fuel type rule: {str(e)}")
    
    # Rule 3: Order number uniqueness
    if 'order_number' in df.columns:
        try:
            duplicates_mask = df.duplicated(subset=['order_number'], keep=False)
            violated_indices = df[duplicates_mask].index
            
            for idx in violated_indices:
                if idx not in all_violations_by_index:
                    all_violations_by_index[idx] = []
                all_violations_by_index[idx].append({
                    'rule_name': 'Uniqueness Constraint',
                    'violation_reason': "Duplicate order_number found",
                    'violated_field': "order_number"
                })
        except Exception as e:
            st.warning(f"Error checking uniqueness rule: {str(e)}")
    
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
            
            # Mark all rows with inconsistent customer IDs as violations
            violated_indices = df[df['customer_id'].isin(inconsistent_customers)].index
            
            for idx in violated_indices:
                if idx not in all_violations_by_index:
                    all_violations_by_index[idx] = []
                all_violations_by_index[idx].append({
                    'rule_name': 'Customer Consistency',
                    'violation_reason': "Same customer_id has different email/name",
                    'violated_field': "customer_id, email, customer_name"
                })
        except Exception as e:
            st.warning(f"Error checking customer consistency rule: {str(e)}")
    
    # Calculate score
    if not all_violations_by_index:
        # No violations found
        return 1.0, pd.DataFrame()
    
    # Get unique violated row indices
    violated_indices = list(all_violations_by_index.keys())
    violations_count = len(violated_indices)
    total_rows = len(df)
    
    # Calculate consistency score: 1.0 = perfect, 0.0 = all bad
    violation_score = 1 - (violations_count / total_rows) if total_rows > 0 else 1.0
    
    # Build violation dataframe by aggregating violations per index
    # FIX: Properly preserve original indices and aggregate violation metadata
    unique_violations = df.loc[violated_indices].copy()
    
    # Aggregate violation details per row
    violation_details_list = []
    for idx in violated_indices:
        violations_for_row = all_violations_by_index[idx]
        
        # Aggregate multiple violations for the same row
        rule_names = ', '.join(set(v['rule_name'] for v in violations_for_row))
        reasons = ' | '.join(v['violation_reason'] for v in violations_for_row)
        fields = ', '.join(set(f for v in violations_for_row for f in v['violated_field'].split(', ')))
        
        violation_details_list.append({
            'index': idx,
            'rule_name': rule_names,
            'violation_reason': reasons,
            'violated_field': fields
        })
    
    violation_details = pd.DataFrame(violation_details_list)
    
    # FIX: Merge using index-based join to preserve row alignment
    unique_violations = unique_violations.merge(
        violation_details.set_index('index'),
        left_index=True,
        right_index=True,
        how='left'
    )
    
    return violation_score, unique_violations

