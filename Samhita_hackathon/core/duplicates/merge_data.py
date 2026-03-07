

import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
@st.cache_data(show_spinner="Merging duplicates...")
def get_combined_merged_data(df):
    fuzzy_merged_count = 0
    exact_merged_count = 0
    total_merged_count = 0
    """
    Merges duplicate customer records based on hierarchical matching rules and
    returns (fuzzy_merged_df, exact_merged_df, all_merged_df).
    
    Rule 1 & 2: Exact Merges (ID match or 2+ field matches)
    Rule 3: Fuzzy Merges (1 exact + fuzzy match)
    """
    original_columns = df.columns.tolist()
    temp_df = df.fillna("").astype(str)
    
    original_rows = temp_df.shape[0]
    temp_df.drop_duplicates(inplace = True)
    rows_after_dedup = temp_df.shape[0]


    temp_df = temp_df.sort_values(by=['customer_id', 'customer_name']).reset_index(drop=True)
    
    records = temp_df.to_dict('records')
    if not records:
        empty_df = pd.DataFrame(columns=original_columns)
        return empty_df, empty_df, empty_df

    all_merged_records = []
    exact_only_records = []
    fuzzy_only_records = []
    consumed_indices = set()
    
    # Look-ahead window to limit comparisons
    window_size = 20

    for i in range(len(records)):
        if i in consumed_indices:
            continue
            
        row_curr = records[i].copy()
        was_exact_merged = False
        was_fuzzy_merged = False
        
        look_ahead_limit = min(i + window_size, len(records)) 
        
        for j in range(i + 1, look_ahead_limit):
            if j in consumed_indices:
                continue
                
            row_next = records[j]
            is_match = False
            current_match_type = None # Track if this specific j-match is exact or fuzzy

            # --- RULE 1: EXACT CUSTOMER ID MATCH ---
            if is_partially_present(row_curr['customer_id'], row_next['customer_id']):
                is_match = True
                current_match_type = 'exact'
            
            # --- RULE 2: 2+ EXACT MATCHES AMONG KEY FIELDS ---
            if not is_match:
                exact_matches = 0
                key_fields = ['customer_name', 'email', 'phone_primary', 'phone_secondary']
                for field in key_fields:
                    if is_partially_present(row_curr[field], row_next[field]):
                        exact_matches += 1
                
                if exact_matches >= 2:
                    is_match = True
                    current_match_type = 'exact'
            
            # --- RULE 3: 1 EXACT + FUZZY MATCH FALLBACK ---
            if not is_match:
                has_any_similarity = any([
                    is_partially_present(row_curr['customer_name'], row_next['customer_name']),
                    is_partially_present(row_curr['email'], row_next['email']),
                    is_partially_present(row_curr['phone_primary'], row_next['phone_primary']),
                    is_partially_present(row_curr['phone_secondary'], row_next['phone_secondary'])
                ])
                
                if has_any_similarity:
                    name_fuzzy = is_fuzzy_match(row_curr['customer_name'], row_next['customer_name'], threshold=90)
                    email_fuzzy = is_fuzzy_match(row_curr['email'], row_next['email'], threshold=95)
                    street_fuzzy = is_fuzzy_match(row_curr['street_address'], row_next['street_address'], threshold=90)
                    
                    if name_fuzzy or email_fuzzy or street_fuzzy:
                        is_match = True
                        current_match_type = 'fuzzy'

            # --- EXECUTE MERGE IF MATCH FOUND ---
            if is_match:
                consumed_indices.add(j)
                if current_match_type == 'exact':
                    was_exact_merged = True
                else:
                    was_fuzzy_merged = True
                
                # Combine all column values
                for col in row_curr:
                    row_curr[col] = clean_merge_values(row_curr[col], row_next[col])
        
        # Add to the master list
        all_merged_records.append(row_curr)
        total_merged_count += 1
        
        # Categorize the merged record
        # Note: If a record hit both exact and fuzzy rules during its window, 
        # it is technically a hybrid, but we prioritize exact categorization or include in both.
        if was_exact_merged:
            exact_merged_count += 1
            exact_only_records.append(row_curr)
        if was_fuzzy_merged:
            fuzzy_merged_count += 1
            fuzzy_only_records.append(row_curr)

    # Convert back to DataFrames
    all_merged_df = pd.DataFrame(all_merged_records)
    exact_merged_df = pd.DataFrame(exact_only_records)
    fuzzy_merged_df = pd.DataFrame(fuzzy_only_records)
    
    # Ensure columns match and handle empty cases
    def finalize_df(df):
        if df.empty:
            return pd.DataFrame(columns=original_columns)
        return df[original_columns].reset_index(drop=True).replace(to_replace="", value=np.nan)
    
    return finalize_df(fuzzy_merged_df), finalize_df(exact_merged_df), finalize_df(all_merged_df)

def is_partially_present(val1, val2):
    """
    Checks if one value is contained within another (bidirectional).
    Used for detecting if merged records share common values.
    Example: "john doe" and "john doe / jane smith" -> True
    """

    v1 = str(val1 or "").strip().lower()
    v2 = str(val2 or "").strip().lower()
    if not v1 or not v2: 
        return False
    return (v1 in v2) or (v2 in v1)


def is_fuzzy_match(val1, val2, threshold=85):
    """
    Performs fuzzy string matching using token sort ratio.
    Returns True if similarity score meets or exceeds threshold.
    """
    if not val1 or not val2: 
        return False
    return fuzz.token_sort_ratio(str(val1), str(val2)) >= threshold


def clean_merge_values(curr_val, next_val):
    """
    Merges two values with '/' separator while avoiding duplicates.
    Handles already merged values intelligently.
    Example: merge("john doe", "john doe / jane smith") -> "john doe / jane smith"
    """
    if not next_val or str(next_val).strip() == "":
        return curr_val
    
    # Split both values by '/' to get individual components
    curr_parts = [p.strip() for p in str(curr_val).split('/') if p.strip()]
    next_parts = [p.strip() for p in str(next_val).split('/') if p.strip()]
    
    # Add new parts only if they don't already exist (case-insensitive)
    for part in next_parts:
        if part.lower() not in [c.lower() for c in curr_parts]:
            curr_parts.append(part)
            
    return " / ".join(curr_parts)
