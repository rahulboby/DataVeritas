import pandas as pd
import numpy as np
from rapidfuzz import fuzz


def get_combined_merged_data(df):
    """
    Merges duplicate customer records based on hierarchical matching rules.
    
    Rule 1 & 2: Exact Merges (ID match or 2+ field matches)
    Rule 3: Fuzzy Merges (1 exact + fuzzy match)
    
    Returns:
        Tuple of (fuzzy_merged_df, exact_merged_df, all_merged_df)
    """
    original_columns = df.columns.tolist()
    temp_df = df.fillna("").astype(str)

    temp_df.drop_duplicates(inplace=True)
    temp_df = temp_df.sort_values(by=['customer_id', 'customer_name']).reset_index(drop=True)

    records = temp_df.to_dict('records')
    if not records:
        empty_df = pd.DataFrame(columns=original_columns)
        return empty_df, empty_df, empty_df

    all_merged_records = []
    exact_only_records = []
    fuzzy_only_records = []
    consumed_indices = set()

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
            current_match_type = None

            # Rule 1: Exact Customer ID Match
            if is_partially_present(row_curr['customer_id'], row_next['customer_id']):
                is_match = True
                current_match_type = 'exact'

            # Rule 2: 2+ exact matches among key fields
            if not is_match:
                exact_matches = 0
                key_fields = ['customer_name', 'email', 'phone_primary', 'phone_secondary']
                for field in key_fields:
                    if field in row_curr and field in row_next:
                        if is_partially_present(row_curr[field], row_next[field]):
                            exact_matches += 1

                if exact_matches >= 2:
                    is_match = True
                    current_match_type = 'exact'

            # Rule 3: 1 exact + fuzzy match fallback
            if not is_match:
                has_any_similarity = any([
                    is_partially_present(row_curr.get('customer_name', ''), row_next.get('customer_name', '')),
                    is_partially_present(row_curr.get('email', ''), row_next.get('email', '')),
                    is_partially_present(row_curr.get('phone_primary', ''), row_next.get('phone_primary', '')),
                    is_partially_present(row_curr.get('phone_secondary', ''), row_next.get('phone_secondary', ''))
                ])

                if has_any_similarity:
                    name_fuzzy = is_fuzzy_match(
                        row_curr.get('customer_name', ''), row_next.get('customer_name', ''), threshold=90
                    )
                    email_fuzzy = is_fuzzy_match(
                        row_curr.get('email', ''), row_next.get('email', ''), threshold=95
                    )
                    street_fuzzy = is_fuzzy_match(
                        row_curr.get('street_address', ''), row_next.get('street_address', ''), threshold=90
                    )

                    if name_fuzzy or email_fuzzy or street_fuzzy:
                        is_match = True
                        current_match_type = 'fuzzy'

            # Execute merge
            if is_match:
                consumed_indices.add(j)
                if current_match_type == 'exact':
                    was_exact_merged = True
                else:
                    was_fuzzy_merged = True

                for col in row_curr:
                    row_curr[col] = clean_merge_values(row_curr[col], row_next[col])

        all_merged_records.append(row_curr)

        if was_exact_merged:
            exact_only_records.append(row_curr)
        if was_fuzzy_merged:
            fuzzy_only_records.append(row_curr)

    # Convert back to DataFrames
    all_merged_df = pd.DataFrame(all_merged_records)
    exact_merged_df = pd.DataFrame(exact_only_records)
    fuzzy_merged_df = pd.DataFrame(fuzzy_only_records)

    def finalize_df(result_df):
        if result_df.empty:
            return pd.DataFrame(columns=original_columns)
        return result_df[original_columns].reset_index(drop=True).replace(to_replace="", value=np.nan)

    return finalize_df(fuzzy_merged_df), finalize_df(exact_merged_df), finalize_df(all_merged_df)


def is_partially_present(val1, val2):
    """Check if one value is contained within another (bidirectional)."""
    v1 = str(val1 or "").strip().lower()
    v2 = str(val2 or "").strip().lower()
    if not v1 or not v2:
        return False
    return (v1 in v2) or (v2 in v1)


def is_fuzzy_match(val1, val2, threshold=85):
    """Performs fuzzy string matching using token sort ratio."""
    if not val1 or not val2:
        return False
    return fuzz.token_sort_ratio(str(val1), str(val2)) >= threshold


def clean_merge_values(curr_val, next_val):
    """Merges two values with '/' separator while avoiding duplicates."""
    if not next_val or str(next_val).strip() == "":
        return curr_val

    curr_parts = [p.strip() for p in str(curr_val).split('/') if p.strip()]
    next_parts = [p.strip() for p in str(next_val).split('/') if p.strip()]

    for part in next_parts:
        if part.lower() not in [c.lower() for c in curr_parts]:
            curr_parts.append(part)

    return " / ".join(curr_parts)


def get_global_exact_duplicates(df):
    """Get all rows that are exact duplicates of another row."""
    dup_mask = df.duplicated(keep=False)
    return df[dup_mask]
