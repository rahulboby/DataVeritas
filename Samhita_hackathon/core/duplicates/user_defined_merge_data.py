import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from typing import List, Dict, Tuple

# ------------------------ cached helpers ------------------------

@st.cache_data
def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to make any edited dataframe consistent with dtypes"""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else str(x))
    return out


def normalize_string(val):
    """Normalize a value to lowercase string for comparison."""
    return str(val or "").strip().lower()


def clean_merge_values(curr_val, next_val):
    """Merge two values with '/' separator, avoiding duplicates."""
    if not next_val or str(next_val).strip() == "":
        return curr_val
    if not curr_val or str(curr_val).strip() == "":
        return next_val

    curr_parts = [p.strip() for p in str(curr_val).split("/") if p.strip()]
    next_parts = [p.strip() for p in str(next_val).split("/") if p.strip()]

    for part in next_parts:
        if part.lower() not in [c.lower() for c in curr_parts]:
            curr_parts.append(part)

    return " / ".join(curr_parts)


# ------------------------ scoring engine ------------------------

def calculate_field_score(val1, val2, logic: str, threshold: int) -> float:
    """
    Calculate similarity score for a single field.
    Returns a score between 0.0 and 1.0
    """
    v1 = normalize_string(val1)
    v2 = normalize_string(val2)
    
    # Empty values
    if not v1 or not v2:
        return 0.0
    
    if logic == "Exact":
        return 1.0 if v1 == v2 else 0.0
    else:  # Fuzzy
        similarity = fuzz.token_sort_ratio(str(val1), str(val2))
        # Normalize to 0-1 scale, considering threshold as minimum
        if similarity >= threshold:
            return similarity / 100.0
        else:
            return 0.0


def calculate_rule_score(row1: pd.Series, row2: pd.Series, rule: dict) -> float:
    """
    Calculate weighted score for a rule between two rows.
    Returns a score between 0.0 and 1.0
    """
    total_score = 0.0
    
    for field in rule.get('fields', []):
        column = field['column']
        logic = field.get('logic', 'Exact')
        threshold = field.get('threshold', 85)
        weight = field.get('weight', 0.0)
        
        field_score = calculate_field_score(
            row1[column], 
            row2[column], 
            logic, 
            threshold
        ) 
        
        total_score += field_score * weight
    
    return total_score


def calculate_total_score(row1: pd.Series, row2: pd.Series, rules: List[dict], rule_indices: List[int]) -> float:
    """
    Calculate total weighted score across multiple rules.
    Returns a score between 0.0 and 1.0
    """
    total_score = 0.0
    
    for rule_idx in rule_indices:
        rule = rules[rule_idx]
        rule_weight = rule.get('rule_weight', 0.0)
        
        rule_score = calculate_rule_score(row1, row2, rule)
        total_score += rule_score * rule_weight
    
    return total_score


@st.cache_data(show_spinner="Finding potential matches...")
def find_merge_candidates(
    df: pd.DataFrame,
    rules: List[dict],
    rule_indices: List[int],
    min_score: float = 0.0
) -> pd.DataFrame:
    """
    Find all potential merge candidates with their scores.
    Returns DataFrame with columns: idx1, idx2, score, [field scores]
    """
    if df is None or df.empty or not rules or not rule_indices:
        return pd.DataFrame()
    
    candidates = []
    n = len(df)
    
    # Progress tracking for large datasets
    total_comparisons = (n * (n - 1)) // 2
    
    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            row1 = df.iloc[i]
            row2 = df.iloc[j]
            
            # Calculate total score
            total_score = calculate_total_score(row1, row2, rules, rule_indices)
            
            if total_score >= min_score:
                candidate = {
                    'idx1': int(i),
                    'idx2': int(j),
                    'total_score': total_score * 100,  # Convert to percentage
                }
                
                # Add individual rule scores for transparency
                for rule_idx in rule_indices:
                    rule = rules[rule_idx]
                    rule_score = calculate_rule_score(row1, row2, rule)
                    candidate[f'rule_{rule_idx}_score'] = rule_score * 100
                
                candidates.append(candidate)
    
    return pd.DataFrame(candidates)

def merge_records(df: pd.DataFrame, merge_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
    """
    Merge records based on approved pairs.
    Returns merged DataFrame.
    """
    if not merge_pairs:
        return df.copy()
    
    # Build merge groups (handle transitive merges)
    merge_groups = {}
    for idx1, idx2 in merge_pairs:
        # Convert to Python int explicitly
        idx1, idx2 = int(idx1), int(idx2)
        
        # Find existing groups
        group1 = merge_groups.get(idx1, {idx1})
        group2 = merge_groups.get(idx2, {idx2})
        
        # Merge groups
        new_group = group1 | group2
        for idx in new_group:
            merge_groups[idx] = new_group
    
    # Get unique groups
    unique_groups = list({frozenset(g) for g in merge_groups.values()})
    
    # Indices that are NOT in any merge group
    all_merged_indices = set()
    for group in unique_groups:
        all_merged_indices.update(group)
    
    unmerged_indices = set(range(len(df))) - all_merged_indices
    
    # Build merged dataframe
    merged_rows = []
    
    # Add merged groups
    for group in unique_groups:
        group_list = sorted([int(idx) for idx in group])  # Convert all to int
        merged_row = df.iloc[group_list[0]].copy()
        
        # Merge all rows in group
        for idx in group_list[1:]:
            for col in df.columns:
                merged_row[col] = clean_merge_values(merged_row[col], df.iloc[idx][col])
        
        merged_rows.append(merged_row)
    
    # Add unmerged rows
    for idx in sorted(unmerged_indices):
        merged_rows.append(df.iloc[int(idx)].copy())  # Convert to int
    
    return pd.DataFrame(merged_rows).reset_index(drop=True)


# ------------------------ validation ------------------------

def validate_field_weights(rule: dict) -> Tuple[bool, str]:
    """Validate that field weights sum to 1.0"""
    if not rule.get('fields'):
        return False, "No fields configured"
    
    total_weight = sum(f.get('weight', 0.0) for f in rule['fields'])
    
    if abs(total_weight - 1.0) > 0.001:  # Allow small floating point errors
        return False, f"Field weights sum to {total_weight:.3f}, must equal 1.0"
    
    return True, ""


def validate_rule_weights(rules: List[dict], rule_indices: List[int]) -> Tuple[bool, str]:
    """Validate that selected rule weights sum to 1.0"""
    if not rule_indices:
        return False, "No rules selected"
    
    total_weight = sum(rules[idx].get('rule_weight', 0.0) for idx in rule_indices)
    
    if abs(total_weight - 1.0) > 0.001:
        return False, f"Selected rule weights sum to {total_weight:.3f}, must equal 1.0"
    
    return True, ""


# ------------------------ UI ------------------------

@st.fragment
def display_merge_data(df: pd.DataFrame):
    """
    Main UI with weighted scoring and manual merge review.
    """
    st.title("🔗 WEIGHTED DATA MERGING TOOL")

    if df is None or df.empty:
        st.info("No data available.")
        return None

    total_rows = len(df)

    # -------------------- session state --------------------
    st.session_state.setdefault("merge_rules", [])
    st.session_state.setdefault("merge_candidates", None)
    st.session_state.setdefault("auto_merged_df", None)
    st.session_state.setdefault("selected_rules_to_run", [])
    st.session_state.setdefault("manual_merge_selections", set())
    st.session_state.setdefault("score_filter", 85.0)
    st.session_state.setdefault("final_merged_df", None)

    # ==================== 1) RULE BUILDER ====================
    st.subheader("📋 Merge Rules Configuration")

    header_left, header_mid, header_right = st.columns([3, 1.3, 1.3], vertical_alignment="center")
    
    with header_left:
        st.caption("Build weighted merge rules. Field weights and rule weights must each sum to 1.0")

    with header_mid:
        if st.button("➕ New Rule", use_container_width=True, type="primary"):
            st.session_state.merge_rules.append({
                "name": f"Rule {len(st.session_state.merge_rules) + 1}",
                "fields": [],
                "rule_weight": 0.0
            })
            st.rerun(scope="fragment")

    with header_right:
        if st.session_state.merge_rules and st.button("🗑️ Clear All", use_container_width=True):
            if st.button("⚠️ Confirm Clear", use_container_width=True, type="secondary"):
                st.session_state.merge_rules = []
                st.session_state.merge_candidates = None
                st.session_state.selected_rules_to_run = []
                st.rerun(scope="fragment")

    if not st.session_state.merge_rules:
        st.info("👆 Click **New Rule** to create your first merge rule.")
        return None

    # Render each rule
    for idx, rule in enumerate(st.session_state.merge_rules):
        with st.expander(f"⚙️ {rule.get('name', f'Rule {idx+1}')}", expanded=True):
            render_rule_config_weighted(df, rule, idx)

    # ==================== 2) RULE SELECTION & VALIDATION ====================
    st.divider()
    st.subheader("🎯 Select Rules to Run")

    # Display rule selection with weights
    rule_selection_data = []
    for idx, rule in enumerate(st.session_state.merge_rules):
        field_weight_sum = sum(f.get('weight', 0.0) for f in rule.get('fields', []))
        field_weight_valid = abs(field_weight_sum - 1.0) < 0.001 if rule.get('fields') else False
        
        rule_selection_data.append({
            "Index": idx,
            "Rule": rule.get('name', f'Rule {idx+1}'),
            "Fields": len(rule.get('fields', [])),
            "Field Weights Sum": f"{field_weight_sum:.3f}",
            "Field Weights ✓": "✅" if field_weight_valid else "❌",
            "Rule Weight": f"{rule.get('rule_weight', 0.0):.3f}"
        })

    rule_df = pd.DataFrame(rule_selection_data)
    
    st.caption("Select rules to run (their weights must sum to 1.0):")
    selection = st.dataframe(
        rule_df.drop(columns=["Index"]),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="rule_selection_table",
    )

    selected_rows = selection.selection.rows if selection.selection else []
    selected_rule_indices = [int(rule_df.iloc[row]["Index"]) for row in selected_rows]
    st.session_state.selected_rules_to_run = selected_rule_indices

    # Validation
    validation_col1, validation_col2, validation_col3 = st.columns(3)
    
    field_weights_valid = all(
        abs(sum(f.get('weight', 0.0) for f in st.session_state.merge_rules[idx].get('fields', [])) - 1.0) < 0.001
        for idx in selected_rule_indices
    ) if selected_rule_indices else False

    rule_weights_valid, rule_weight_msg = validate_rule_weights(
        st.session_state.merge_rules, 
        selected_rule_indices
    )

    with validation_col1:
        if selected_rule_indices:
            st.metric("Selected Rules", len(selected_rule_indices))
        else:
            st.warning("No rules selected")

    with validation_col2:
        if field_weights_valid:
            st.success("✅ Field weights valid")
        elif selected_rule_indices:
            st.error("❌ Fix field weights")

    with validation_col3:
        if rule_weights_valid:
            st.success("✅ Rule weights valid")
        elif selected_rule_indices:
            st.error(f"❌ {rule_weight_msg}")

    can_run = bool(selected_rule_indices) and field_weights_valid and rule_weights_valid

    # ==================== 3) RUN ANALYSIS ====================
    st.divider()
    st.subheader("▶️ Run Merge Analysis")

    run_col1, run_col2, run_col3 = st.columns([2, 1, 1])
    
    with run_col1:
        run_button = st.button(
            "🔍 Find Merge Candidates",
            type="primary",
            use_container_width=True,
            disabled=not can_run
        )
    
    with run_col2:
        min_score_filter = st.number_input(
            "Min Score %",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Only find pairs above this score"
        )
    
    with run_col3:
        st.metric("Total Rows", f"{total_rows:,}")

    if not can_run:
        st.warning("⚠️ Fix validation errors before running")
        return None

    if run_button:
        with st.spinner("Analyzing potential merges..."):
            candidates_df = find_merge_candidates(
                df=df,
                rules=st.session_state.merge_rules,
                rule_indices=selected_rule_indices,
                min_score=min_score_filter / 100.0
            )
            
            st.session_state.merge_candidates = candidates_df
            
            # Auto-merge 100% matches
            auto_merge_pairs = [
                (int(row['idx1']), int(row['idx2'])) 
                for _, row in candidates_df.iterrows() 
                if row['total_score'] >= 99.99
            ]
            
            if auto_merge_pairs:
                st.session_state.auto_merged_df = merge_records(df, auto_merge_pairs)
            else:
                st.session_state.auto_merged_df = df.copy()
            
            st.session_state.manual_merge_selections = set()
            st.session_state.final_merged_df = None
            
        st.success(f"✅ Found {len(candidates_df):,} potential merge pairs")
        st.rerun(scope="fragment")

    # ==================== 4) RESULTS ====================
    if st.session_state.merge_candidates is None:
        st.info("👆 Configure rules and click **Find Merge Candidates** to begin")
        return None

    candidates_df = st.session_state.merge_candidates
    
    st.divider()
    st.subheader("📊 Merge Results")

    # Statistics
    auto_merge_count = len([1 for _, row in candidates_df.iterrows() if row['total_score'] >= 99.99])
    manual_review_count = len(candidates_df) - auto_merge_count
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Total Candidates", f"{len(candidates_df):,}")
    stat_col2.metric("Auto-Merged (100%)", f"{auto_merge_count:,}", delta="✅ Merged")
    stat_col3.metric("Manual Review", f"{manual_review_count:,}", delta="⏳ Pending")
    stat_col4.metric("After Auto-Merge", f"{len(st.session_state.auto_merged_df):,}")

    # ==================== 5) MANUAL REVIEW ====================
    if manual_review_count > 0:
        st.divider()
        st.subheader("👀 Manual Merge Review")
        
        # Score filter
        filter_col1, filter_col2 = st.columns([3, 1])
        
        with filter_col1:
            score_threshold = st.slider(
                "Filter by Score Threshold (%)",
                min_value=0.0,
                max_value=99.9,
                value=st.session_state.score_filter,
                step=0.1,
                help="Only show pairs with scores above this threshold"
            )
            st.session_state.score_filter = score_threshold
        
        with filter_col2:
            st.metric("Pairs Shown", 
                     len([1 for _, row in candidates_df.iterrows() 
                          if score_threshold <= row['total_score'] < 100]))

        # Filter candidates
        review_candidates = candidates_df[
            (candidates_df['total_score'] >= score_threshold) & 
            (candidates_df['total_score'] < 100)
        ].sort_values('total_score', ascending=False).reset_index(drop=True)

        if len(review_candidates) == 0:
            st.info(f"No pairs found between {score_threshold:.1f}% and 100%")
        else:
            st.caption(f"Showing {len(review_candidates):,} pairs for review")
            
            # Display pairs for review
            render_manual_review_ui(df, review_candidates, st.session_state.merge_rules, selected_rule_indices)

    # ==================== 6) FINALIZE MERGE ====================
    st.divider()
    st.subheader("✅ Finalize Merge")

    finalize_col1, finalize_col2, finalize_col3 = st.columns([2, 1, 1])
    
    with finalize_col1:
        manual_selections = st.session_state.manual_merge_selections
        st.metric("Manual Selections", len(manual_selections))
    
    with finalize_col2:
        total_merges = auto_merge_count + len(manual_selections)
        st.metric("Total Merges", total_merges)
    
    with finalize_col3:
        expected_final = total_rows - total_merges
        st.metric("Expected Final Rows", f"{expected_final:,}")

    if st.button("🔗 Apply Selected Merges", type="primary", use_container_width=True):
        with st.spinner("Merging records..."):
            # Combine auto and manual merges
            all_merge_pairs = [
                (int(row['idx1']), int(row['idx2'])) 
                for _, row in candidates_df.iterrows() 
                if row['total_score'] >= 99.99
            ]
            
            # Add manual selections
            for pair_key in manual_selections:
                idx1, idx2 = map(int, pair_key.split('_'))
                all_merge_pairs.append((idx1, idx2))
            
            final_df = merge_records(df, all_merge_pairs)
            st.session_state.final_merged_df = final_df
        
        st.success(f"✅ Merged {len(all_merge_pairs):,} pairs → Final: {len(final_df):,} rows")
        st.rerun(scope="fragment")

    # ==================== 7) DOWNLOAD ====================
    if st.session_state.final_merged_df is not None:
        st.divider()
        st.subheader("💾 Download Results")
        
        final_df = st.session_state.final_merged_df
        
        preview_col, download_col = st.columns([3, 1])
        
        with preview_col:
            st.caption(f"Preview (showing first 1000 of {len(final_df):,} rows):")
            st.dataframe(
                make_arrow_safe(final_df.head(1000)),
                use_container_width=True,
                hide_index=True
            )
        
        with download_col:
            st.download_button(
                "📥 Download Merged CSV",
                data=final_df.to_csv(index=False).encode("utf-8"),
                file_name="merged_data.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            
            st.metric("Reduction", f"{((total_rows - len(final_df)) / total_rows * 100):.1f}%")

    return st.session_state.final_merged_df


@st.fragment
def render_manual_review_ui(df: pd.DataFrame, candidates: pd.DataFrame, rules: List[dict], rule_indices: List[int]):
    """Render interactive manual review interface for merge candidates"""
    
    for idx, row in candidates.iterrows():
        idx1, idx2 = int(row['idx1']), int(row['idx2'])
        score = row['total_score']
        
        pair_key = f"{idx1}_{idx2}"
        is_selected = pair_key in st.session_state.manual_merge_selections
        
        # Color code by score
        if score >= 95:
            border_color = "🟢"
        elif score >= 85:
            border_color = "🟡"
        else:
            border_color = "🟠"
        
        with st.container(border=True):
            # Header
            header_col1, header_col2, header_col3 = st.columns([0.5, 3, 1])
            
            with header_col1:
                merge_checkbox = st.checkbox(
                    "Merge",
                    value=is_selected,
                    key=f"merge_check_{pair_key}",
                    label_visibility="collapsed"
                )
                
                if merge_checkbox and not is_selected:
                    st.session_state.manual_merge_selections.add(pair_key)
                    st.rerun(scope="fragment")
                elif not merge_checkbox and is_selected:
                    st.session_state.manual_merge_selections.discard(pair_key)
                    st.rerun(scope="fragment")
            
            with header_col2:
                st.markdown(f"{border_color} **Pair {idx + 1}** - Score: **{score:.2f}%**")
            
            with header_col3:
                # Show individual rule scores
                rule_scores_text = " | ".join([
                    f"R{rule_idx+1}: {row.get(f'rule_{rule_idx}_score', 0):.1f}%"
                    for rule_idx in rule_indices
                ])
                st.caption(rule_scores_text)
            
            # Show comparison
            record1 = df.iloc[idx1]
            record2 = df.iloc[idx2]
            
            # Get all columns used in rules
            columns_to_show = set()
            for rule_idx in rule_indices:
                for field in rules[rule_idx].get('fields', []):
                    columns_to_show.add(field['column'])
            
            columns_to_show = sorted(list(columns_to_show))
            
            comparison_data = []
            for col in columns_to_show:
                val1 = str(record1[col]) if pd.notna(record1[col]) else ""
                val2 = str(record2[col]) if pd.notna(record2[col]) else ""
                
                match_icon = "✅" if normalize_string(val1) == normalize_string(val2) else "⚠️"
                
                comparison_data.append({
                    "Field": col,
                    "Record 1": val1[:50] + ("..." if len(val1) > 50 else ""),
                    "Record 2": val2[:50] + ("..." if len(val2) > 50 else ""),
                    "Match": match_icon
                })
            
            st.dataframe(
                pd.DataFrame(comparison_data),
                use_container_width=True,
                hide_index=True,
                height=min(len(comparison_data) * 35 + 38, 300)
            )


@st.fragment
def render_rule_config_weighted(df: pd.DataFrame, rule: dict, idx: int):
    """Render weighted rule configuration UI"""
    
    all_columns = df.columns.tolist()
    
    if 'fields' not in rule:
        rule['fields'] = []
    
    # Rule header
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        rule_name = st.text_input(
            "Rule Name",
            value=rule.get('name', f"Rule {idx + 1}"),
            key=f"rule_name_{idx}",
            placeholder="e.g., Email + Phone Match"
        )
        rule['name'] = rule_name
    
    with col2:
        rule_weight = st.number_input(
            "Rule Weight",
            min_value=0.0,
            max_value=1.0,
            value=rule.get('rule_weight', 0.0),
            step=0.05,
            format="%.3f",
            key=f"rule_weight_{idx}",
            help="Weight of this rule (all selected rules must sum to 1.0)"
        )
        rule['rule_weight'] = rule_weight
    
    with col3:
        if st.button("📋 Duplicate", key=f"dup_rule_{idx}", use_container_width=True):
            duplicated = {
                'name': f"{rule['name']} (Copy)",
                'fields': [f.copy() for f in rule.get('fields', [])],
                'rule_weight': 0.0
            }
            st.session_state.merge_rules.insert(idx + 1, duplicated)
            st.rerun(scope="fragment")
    
    with col4:
        if st.button("🗑️ Delete", key=f"del_rule_{idx}", use_container_width=True, type="secondary"):
            st.session_state.merge_rules.pop(idx)
            st.rerun(scope="fragment")

    st.divider()

    # Field configuration
    st.markdown("##### 🎯 Field Configuration")
    
    # Field weight validation
    total_field_weight = sum(f.get('weight', 0.0) for f in rule['fields'])
    
    weight_col1, weight_col2 = st.columns([3, 1])
    with weight_col1:
        if rule['fields']:
            if abs(total_field_weight - 1.0) < 0.001:
                st.success(f"✅ Field weights sum to 1.0")
            else:
                st.error(f"❌ Field weights sum to {total_field_weight:.3f} (must equal 1.0)")
    
    with weight_col2:
        st.metric("Fields", len(rule['fields']))

    # Render fields
    fields_to_remove = []
    for field_idx, field in enumerate(rule['fields']):
        should_remove = render_field_config_weighted(df, rule, field, field_idx, idx, all_columns)
        if should_remove:
            fields_to_remove.append(field_idx)
    
    for field_idx in sorted(fields_to_remove, reverse=True):
        rule['fields'].pop(field_idx)
        st.rerun(scope="fragment")

    # Add field button
    st.divider()
    used_columns = [f['column'] for f in rule['fields']]
    available_columns = [c for c in all_columns if c not in used_columns]
    
    add_col1, add_col2, add_col3 = st.columns([1, 2, 1])
    with add_col2:
        if st.button(
            f"➕ Add Field ({len(available_columns)} available)",
            key=f"add_field_{idx}",
            use_container_width=True,
            type="primary",
            disabled=len(available_columns) == 0
        ):
            remaining_weight = max(0.0, 1.0 - total_field_weight)
            rule['fields'].append({
                'column': available_columns[0],
                'logic': 'Fuzzy',
                'threshold': 85,
                'weight': round(remaining_weight, 3)
            })
            st.rerun(scope="fragment")


def render_field_config_weighted(df: pd.DataFrame, rule: dict, field: dict, field_idx: int, rule_idx: int, all_columns: list) -> bool:
    """Render weighted field configuration. Returns True if should be removed."""
    
    with st.container(border=True):
        header_col1, header_col2, header_col3 = st.columns([2, 2, 1])
        
        with header_col1:
            st.markdown(f"**Field {field_idx + 1}:** `{field['column']}`")
        
        with header_col2:
            weight = st.number_input(
                "Weight",
                min_value=0.0,
                max_value=1.0,
                value=field.get('weight', 0.0),
                step=0.05,
                format="%.3f",
                key=f"weight_{rule_idx}_{field_idx}",
                help="Weight of this field within the rule"
            )
            field['weight'] = weight
        
        with header_col3:
            if st.button("✖️", key=f"remove_{rule_idx}_{field_idx}", use_container_width=True):
                return True
        
                # Configuration
        config_col1, config_col2, config_col3 = st.columns([2, 1.5, 2])
        
        with config_col1:
            # Column selection
            used_columns = [f['column'] for f_idx, f in enumerate(rule['fields']) if f_idx != field_idx]
            available_columns = [c for c in all_columns if c not in used_columns]
            
            if field['column'] not in available_columns:
                available_columns.insert(0, field['column'])
            
            current_index = available_columns.index(field['column']) if field['column'] in available_columns else 0
            
            selected_column = st.selectbox(
                "Column",
                options=available_columns,
                index=current_index,
                key=f"col_{rule_idx}_{field_idx}",
                label_visibility="visible"
            )
            field['column'] = selected_column
        
        with config_col2:
            logic_index = 0 if field.get('logic', 'Exact') == 'Exact' else 1
            selected_logic = st.radio(
                "Match Logic",
                options=['Exact', 'Fuzzy'],
                index=logic_index,
                key=f"logic_{rule_idx}_{field_idx}",
                horizontal=False
            )
            
            if selected_logic != field['logic']:
                field['logic'] = selected_logic
                st.rerun(scope='fragment')
        
        with config_col3:
            if field['logic'] == 'Fuzzy':
                threshold_val = st.slider(
                    "Similarity Threshold (%)",
                    min_value=0,
                    max_value=100,
                    value=field.get('threshold', 85),
                    step=5,
                    key=f"threshold_{rule_idx}_{field_idx}",
                    help="Minimum similarity percentage for this field",
                    format="%d%%"
                )
                field['threshold'] = threshold_val
                
                # Visual indicator
                if threshold_val >= 95:
                    st.caption("🔒 Very Strict")
                elif threshold_val >= 85:
                    st.caption("⚖️ Balanced")
                elif threshold_val >= 70:
                    st.caption("🎯 Relaxed")
                else:
                    st.caption("⚠️ Very Relaxed")
            else:
                field['threshold'] = 100
                st.info("Exact match required\n(100%)")
        
        # Preview section
        with st.expander(f"📊 Preview: {field['column']}", expanded=False):
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                st.caption("**Sample Values:**")
                sample_values = df[field['column']].dropna().head(5)
                if len(sample_values) > 0:
                    for val in sample_values:
                        st.caption(f"• {str(val)[:50]}")
                else:
                    st.caption("_No data available_")
            
            with preview_col2:
                st.caption("**Statistics:**")
                total = len(df)
                non_null = df[field['column']].notna().sum()
                unique = df[field['column']].nunique()
                
                st.caption(f"Total: {total:,}")
                st.caption(f"Non-null: {non_null:,} ({non_null/total*100:.1f}%)")
                st.caption(f"Unique: {unique:,}")
                
                if unique == total:
                    st.warning("⚠️ All unique - may not merge many records")
                elif unique < 10:
                    st.info("ℹ️ Low cardinality - good for grouping")
    
    return False

