import streamlit as st
import pandas as pd
from core.consistency.rule_engine import apply_rules
from core.downloads import data_downloader as DH


# ------------------------ RULE BUILDER UI ------------------------

@st.fragment
def render_rule_builder(df):
    """Dynamic rule builder interface."""
    
    st.subheader("Rule Configuration")
    
    # Initialize rules in session state
    if 'consistency_rules' not in st.session_state:
        st.session_state.consistency_rules = []
    
    st.divider()
    
    # Display existing rules
    if st.session_state.consistency_rules:
        st.markdown("### Active Rules")
        
        rules_to_remove = []
        
        for idx, rule in enumerate(st.session_state.consistency_rules):
            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{rule['name']}**")
                    st.caption(f"Type: {rule['rule_type']} | Field: {rule.get('field', 'N/A')}")
                
                with col2:
                    rule['enabled'] = st.toggle(
                        "Enabled",
                        value=rule.get('enabled', True),
                        key=f"toggle_{rule['id']}"
                    )
                
                with col3:
                    if st.button("Delete", key=f"delete_{rule['id']}", use_container_width=True):
                        rules_to_remove.append(idx)
                
                # Expandable rule configuration
                with st.expander("Edit Rule", expanded=False):
                    render_rule_config(df, rule, idx)
        
        # Remove deleted rules
        for idx in sorted(rules_to_remove, reverse=True):
            st.session_state.consistency_rules.pop(idx)
            st.rerun(scope="fragment")  # Force fragment rerun
    else:
        st.info("No rules configured. Click 'Add New Rule' to get started.")

    # Button to add new rule
    if st.button("Add New Rule", type="primary", use_container_width=True):
        # Generate unique ID based on timestamp to avoid conflicts
        import time
        new_id = int(time.time() * 1000000) % 1000000  # Microsecond-based unique ID
        
        st.session_state.consistency_rules.append({
            'id': new_id,
            'name': f"Rule {len(st.session_state.consistency_rules) + 1}",
            'enabled': True,
            'rule_type': 'Mandatory Field',
            'field': df.columns[0] if len(df.columns) > 0 else None
        })
        st.rerun(scope="fragment")  # Force fragment rerun immediately



def render_rule_config(df, rule, idx):
    """Render configuration UI for a specific rule."""
    
    all_columns = df.columns.tolist()
    
    # Rule Name
    rule['name'] = st.text_input(
        "Rule Name",
        value=rule.get('name', f"Rule {idx + 1}"),
        key=f"name_{rule['id']}"
    )
    
    # Rule Type
    rule_types = [
        "Mandatory Field",
        "Data Type",
        "Format",
        "Range",
        "Uniqueness",
        "Cross-Field Comparison",
        "Conditional (IF-THEN)"
    ]
    
    rule['rule_type'] = st.selectbox(
        "Rule Type",
        rule_types,
        index=rule_types.index(rule.get('rule_type', 'Mandatory Field')),
        key=f"type_{rule['id']}"
    )
    
    # Dynamic parameters based on rule type
    if rule['rule_type'] in ["Mandatory Field", "Data Type", "Format", "Range", "Uniqueness"]:
        rule['field'] = st.selectbox(
            "Field",
            all_columns,
            index=all_columns.index(rule.get('field', all_columns[0])) if rule.get('field') in all_columns else 0,
            key=f"field_{rule['id']}"
        )
    
    # Type-specific parameters
    if rule['rule_type'] == "Data Type":
        rule['expected_type'] = st.selectbox(
            "Expected Type",
            ["String", "Integer", "Float", "Date", "Boolean"],
            key=f"dtype_{rule['id']}"
        )
    
    elif rule['rule_type'] == "Format":
        rule['format_type'] = st.selectbox(
            "Format Pattern",
            ["Email", "Phone", "URL", "Zipcode", "Custom"],
            key=f"format_{rule['id']}"
        )
        if rule['format_type'] == "Custom":
            rule['custom_pattern'] = st.text_input(
                "Regex Pattern",
                value=rule.get('custom_pattern', ''),
                key=f"pattern_{rule['id']}"
            )
    
    elif rule['rule_type'] == "Range":
        c1, c2 = st.columns(2)
        with c1:
            rule['min_value'] = st.number_input(
                "Min Value",
                value=rule.get('min_value'),
                key=f"min_{rule['id']}"
            )
        with c2:
            rule['max_value'] = st.number_input(
                "Max Value",
                value=rule.get('max_value'),
                key=f"max_{rule['id']}"
            )
    
    elif rule['rule_type'] == "Cross-Field Comparison":
        c1, c2 = st.columns(2)
        with c1:
            rule['field'] = st.selectbox(
                "Field 1",
                all_columns,
                key=f"field1_{rule['id']}"
            )
        with c2:
            rule['field2'] = st.selectbox(
                "Field 2",
                all_columns,
                key=f"field2_{rule['id']}"
            )
        
        rule['operator'] = st.selectbox(
            "Operator",
            ["<", ">", "<=", ">=", "==", "!="],
            key=f"op_{rule['id']}"
        )
    
    elif rule['rule_type'] == "Conditional (IF-THEN)":
        st.markdown("**IF** condition:")
        c1, c2 = st.columns(2)
        with c1:
            rule['field'] = st.selectbox(
                "Condition Field",
                all_columns,
                key=f"cond_field_{rule['id']}"
            )
        with c2:
            rule['condition_value'] = st.text_input(
                "Equals Value",
                value=rule.get('condition_value', ''),
                key=f"cond_val_{rule['id']}"
            )
        
        st.markdown("**THEN** requirement:")
        c1, c2 = st.columns(2)
        with c1:
            rule['then_field'] = st.selectbox(
                "Then Field",
                all_columns,
                key=f"then_field_{rule['id']}"
            )
        with c2:
            rule['then_value'] = st.text_input(
                "Must Equal",
                value=rule.get('then_value', ''),
                key=f"then_val_{rule['id']}"
            )


# ------------------------ MAIN DISPLAY FUNCTION ------------------------

def displayConsistencyStats(df):
    
    # Rule Builder Section
    render_rule_builder(df)
    
    st.divider()
    
    # Validate Button
    if st.button("Run Validation", type="primary", use_container_width=True):
        if not st.session_state.get('consistency_rules'):
            st.warning("No rules configured. Please add at least one rule.")
            return
        
        with st.spinner("Validating data against rules..."):
            # Apply rules
            violation_df, violation_summary = apply_rules(
                df,
                st.session_state.consistency_rules
            )
            
            # Store results in session state
            st.session_state['validation_results'] = {
                'violations': violation_df,
                'summary': violation_summary
            }
        
        st.success("Validation complete.")
    
    # Display Results
    if 'validation_results' in st.session_state:
        violation_df = st.session_state['validation_results']['violations']
        violation_summary = st.session_state['validation_results']['summary']
        
        total_rows = len(df)
        violations_count = len(violation_df.drop_duplicates(subset=df.columns.tolist(), keep='first')) if not violation_df.empty else 0
        
        # Calculate consistency score
        violation_score = 1 - (violations_count / total_rows) if total_rows > 0 else 1.0
        percent_bad = 1 - violation_score
        
        # Metrics
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{total_rows:,}")
        
        with col2:
            st.metric(
                "Violations Found",
                f"{violations_count:,}",
                delta=f"{percent_bad:.1%} of total" if violations_count > 0 else None,
                delta_color="inverse"
            )
        
        with col3:
            st.metric("Consistency Score", f"{violation_score:.1%}")
        
        # Visual Health Bar
        col_left, col_right = st.columns([1, 4])
        
        with col_left:
            st.write("**Overall Data Health**")
            if violation_score >= 0.95:
                st.success(f"Excellent! {violation_score:.1%} compliant")
            elif violation_score >= 0.80:
                st.warning(f"Good, but {violations_count} violations found")
            else:
                st.error(f"Critical: {percent_bad:.1%} inconsistent")
            
            st.progress(violation_score)
        
        with col_right:
            # Rule-wise breakdown with interactive selection
            st.write("**Rule-Wise Violation Summary**")
            
            # Create interactive dataframe
            summary_df = pd.DataFrame([
                {"Rule": k, "Violations": v}
                for k, v in violation_summary.items()
            ]).sort_values("Violations", ascending=False)
            
            # Use data editor for selection
            edited_df = st.data_editor(
                summary_df,
                column_config={
                    "Rule": st.column_config.TextColumn("Rule Name", width="large"),
                    "Violations": st.column_config.NumberColumn("Violations", width="medium")
                },
                disabled=["Rule", "Violations"],
                hide_index=True,
                use_container_width=True,
                key="rule_selector"
            )
            
            # Alternative: Use multiselect for rule filtering
            st.caption("**Filter by specific rules:**")
            selected_rules = st.multiselect(
                "Select rules to filter violations",
                options=list(violation_summary.keys()),
                default=[],
                key="rule_filter",
                help="Leave empty to show all violations"
            )
        
        st.divider()
        
        # Filter violations based on selected rules
        if violations_count > 0:
            # Filter violation_df based on selected rules
            if selected_rules:
                filtered_violation_df = violation_df[violation_df['rule_name'].isin(selected_rules)]
                violated_indices = filtered_violation_df.index.unique()
                filter_info = f" (Filtered by: {', '.join(selected_rules)})"
            else:
                violated_indices = violation_df.index.unique()
                filter_info = " (All rules)"
            
            consistent_df = df[~df.index.isin(violated_indices)].copy()
            inconsistent_df = df[df.index.isin(violated_indices)].copy()
            
            # Add violation details to inconsistent records
            if selected_rules:
                # Only show violations for selected rules
                relevant_violations = violation_df[violation_df['rule_name'].isin(selected_rules)]
            else:
                relevant_violations = violation_df
            
            if not relevant_violations.empty:
                # Aggregate violation reasons per row
                violation_details = relevant_violations.groupby(relevant_violations.index).agg({
                    'violation_reason': lambda x: ' | '.join(x),
                    'violated_field': lambda x: ', '.join(set(str(v) for v in x)),
                    'rule_name': lambda x: ', '.join(set(x))
                }).reset_index()
                
                inconsistent_df = inconsistent_df.merge(
                    violation_details,
                    left_index=True,
                    right_on='index',
                    how='left'
                )
            
            # Tabs for consistent/inconsistent
            tab1, tab2 = st.tabs([
                f"Consistent Records ({len(consistent_df):,}){filter_info}",
                f"Inconsistent Records ({len(inconsistent_df):,}){filter_info}"
            ])
            
            with tab1:
                if len(consistent_df) > 0:
                    st.dataframe(consistent_df.head(10000), use_container_width=True)
                    DH.add_download_buttons(consistent_df, f"consistent_records{'_filtered' if selected_rules else ''}", show_header=False)
                else:
                    st.info("No consistent records found for the selected filter.")
            
            with tab2:
                if len(inconsistent_df) > 0:
                    # Highlight violated fields on a per-row basis
                    def highlight_row_violations(row):
                        """Highlight only the specific violated fields for each row."""
                        violated_field_value = row.get('violated_field', '')
                        
                        if pd.isna(violated_field_value) or violated_field_value == '':
                            return ['' for _ in row.index]
                        
                        violated_fields = [f.strip() for f in str(violated_field_value).split(',')]
                        
                        return ['background-color: #FFCDD2' if col_name in violated_fields else '' 
                                for col_name in row.index]
                    
                    # Calculate cells to display
                    display_limit = 10000
                    cells_to_render = min(len(inconsistent_df), display_limit) * len(inconsistent_df.columns)
                    
                    # Only apply styling if under reasonable limit
                    if cells_to_render <= 100000:
                        pd.set_option("styler.render.max_elements", cells_to_render + 10000)
                        st.dataframe(
                            inconsistent_df.head(display_limit).style.apply(highlight_row_violations, axis=1),
                            use_container_width=True
                        )
                    else:
                        st.warning(f"Dataset too large to highlight ({cells_to_render:,} cells). Showing without styling.")
                        st.dataframe(
                            inconsistent_df.head(display_limit),
                            use_container_width=True
                        )
                        
                        # Show which columns are commonly violated
                        with st.expander("Most Frequently Violated Fields"):
                            if 'violated_field' in inconsistent_df.columns:
                                all_violations = inconsistent_df['violated_field'].dropna().str.split(', ').explode()
                                violation_counts = all_violations.value_counts().head(10)
                                st.bar_chart(violation_counts)
                    
                    DH.add_download_buttons(inconsistent_df, f"inconsistent_records{'_filtered' if selected_rules else ''}", show_header=False)
                else:
                    st.info("No inconsistent records found for the selected filter.")
        
        else:
            st.success("No inconsistencies detected. All records are compliant.")
            DH.add_download_buttons(df, "validated_data", show_header=True)

