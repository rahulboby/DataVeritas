import pandas as pd
import plotly.express as px
import streamlit as st
from rapidfuzz import fuzz
import numpy as np
from core.duplicates.user_defined_merge_data import display_merge_data
from core.duplicates.global_exact_duplicates import getGlobalExactDuplicates
def displayDuplicateStats(df):
    # --- 0. CUSTOM CSS FOR PROFESSIONAL THEME ---
    st.markdown("""
        <style>
        .main-title {
            font-size: 3rem !important;
            font-weight: 700;
            color: #1E3A8A;
            margin-top: -20px;
            margin-bottom: 20px;
        }
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)


    # --- 1. CALCULATIONS ---
    total_rows = df.shape[0]
    
    # Rows involved in duplication (e.g., the 1,600)
    dup_mask = df.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    
    # Rows that are 100% unique (e.g., 18,400)
    pure_unique_count = total_rows - dup_count
    
    # Rows to remove (e.g., 1,200)
    
    #Filter out duplicates from [name, cid, email, phone]
    deduplicated_df_columnset =  df.drop_duplicates(keep = 'first')

    deduplicated_count = deduplicated_df_columnset.shape[0]
    redundant_rows_to_remove = total_rows - deduplicated_count
    
    # Unique rows that have duplicates (e.g., 400)
    unique_with_duplicates = deduplicated_count - pure_unique_count

    # --- 2. EXACT ROW DUPLICATES (GLOBAL) ---
    with st.expander("Global Exact Duplicates (All Columns Match)", expanded=False):
        styled_df = getGlobalExactDuplicates(df, dup_count, dup_mask)

        if styled_df is not None:
            st.dataframe(styled_df)
        else:
            st.info("Analysis complete: No exact row-level duplicates identified.", )

    # Metric for Over all duplicates
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{total_rows:,}", help="Total number of records in the dataset.")
    col2.metric("Purely Unique Records", f"{pure_unique_count:,}", help="Records with 100% uniqueness across all columns.")
    col3.metric("Unique Rows (w/ Duplicates)", f"{unique_with_duplicates:,}", help="Unique records that have duplicates elsewhere in the dataset.")
    col4.metric("Redundant Records", f"{redundant_rows_to_remove:,}", help="Total number of records that can be removed to eliminate duplicates based on key fields.")

    # --- 3. VISUAL ANALYTICS ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Top 5 Columns by Duplicate Density", help = "Columns with the highest number of duplicate values.")
        rows = []
        for col in df.columns:
            dup_vals = df[col].duplicated(keep=False).sum()
            rows.append({"column": col, "duplicate_count": int(dup_vals)})
        bar_df = pd.DataFrame(rows).sort_values("duplicate_count", ascending=False).head(5)
        if not bar_df.empty:
            fig_bar = px.bar(bar_df, x="duplicate_count", y="column", orientation="h", color_discrete_sequence=["#B20202"])
            fig_bar.update_xaxes(range = [0, df.shape[0]])
            fig_bar.update_layout(xaxis_title="Count of Duplicate Values", yaxis_title="", height=400, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_bar)
        else:
            st.info("No field-level duplicates detected. All columns have 100% Cardinality")
        

    with col_right:
        st.subheader("Bottom 5 Columns by Duplicate Density", help = "Columns with the lowest number of duplicate values.")
        rows = []  
        for col in df.columns:
            dup_vals = df[col].duplicated(keep=False).sum()
            rows.append({"column": col, "duplicate_count": int(dup_vals)})
        bar_df = pd.DataFrame(rows).sort_values("duplicate_count", ascending=True).head(5)
        if not bar_df.empty:
            fig_bar = px.bar(bar_df, x="duplicate_count", y="column", orientation="h", color_discrete_sequence=["#5185D4"])
            fig_bar.update_xaxes(range = [0, df.shape[0]])
            fig_bar.update_layout(xaxis_title="Count of Duplicate Values", yaxis_title="", height=400, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_bar)
        else:
            st.info("No field-level duplicates detected. All columns have 100% Cardinality")

    with st.expander("Show All Columns by Duplicate Density", expanded=False):
        rows = []
        for col in df.columns:
            dup_vals = df[col].duplicated(keep=False).sum()
            rows.append({"column": col, "duplicate_count": int(dup_vals)})
        bar_df = pd.DataFrame(rows).sort_values("duplicate_count", ascending=True)
        dup_cols = bar_df[bar_df["duplicate_count"] > 0]
        non_dup_cols = bar_df[bar_df["duplicate_count"] == 0]

        if not dup_cols.empty:
            fig_bar = px.bar(dup_cols, x="duplicate_count", y="column", orientation="h", color_discrete_sequence=["#C0392B"])
            fig_bar.update_layout(xaxis_title="Count of Duplicate Values", yaxis_title="",  height = 1000, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_bar)
            if not non_dup_cols.empty:
                with st.expander("Columns without any duplicates: "):
                    st.write(non_dup_cols)
            else:
                st.info("No columns without duplicates")

        else:
            st.info("No field-level duplicates detected. All columns have 100% Cardinality")


    # --- 4. FIELD-SPECIFIC Duplicate Explorer ---
    displayDuplicateExplorer(df)

    # --------------------- DATA MERGING ---------------------

    # 1. Custom CSS to define the "highlight" look
    st.markdown("""
        <style>
        /* Style for the highlighted container */
        [data-testid="stVerticalBlockBorderWrapper"].highlighted {
            background-color: rgba(28, 131, 225, 0.1); /* Light blue background */
            border-color: #1C83E1 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    


# PART 1: Cached expensive operations
@st.cache_data
def calculate_frequency(df, column_name):
    """
    Cached frequency calculation - only recalculates when df or column changes.
    """
    freq_df = df[column_name].value_counts().reset_index()
    freq_df.columns = [column_name, "Occurrence Count"]
    return freq_df


@st.cache_data
def filter_by_value(df, column_name, selected_value):
    """
    Cached filtering operation - only recalculates when parameters change.
    """
    return df[df[column_name] == selected_value]


# PART 2: UI with fragment
@st.fragment
def displayDuplicateExplorer(df):
    st.subheader("Column Duplicate Explorer")
    
    # 1. Initialize session state
    if "dup_selected_value" not in st.session_state:
        st.session_state["dup_selected_value"] = None
    
    # 2. Column Selection
    all_cols = df.columns.tolist()
    target_col = st.selectbox("Select Column to Group By:", all_cols, key="dup_col_select")
    
    # 3. Calculate frequency using CACHED function (fast!)
    freq_df = calculate_frequency(df, target_col)
    
    # 4. Layout: Two Columns
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.write(f"**Unique Values in {target_col}:**")
        
        # Use selection_mode="single_row" to capture clicks
        selection = st.dataframe(
            freq_df,
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            width='stretch',
            key="freq_table"
        )
        st.caption("Click a row to filter the table")

    with col_right:
        # 5. Check if a row is selected
        selected_rows = selection.selection.rows
        
        if selected_rows:
            # Get the actual value from the clicked row
            selected_index = selected_rows[0]
            selected_value = freq_df.iloc[selected_index][target_col]
            
            st.write(f"**Records where {target_col} is:** `{selected_value}`")
            
            # Use CACHED filtering (fast!)
            filtered_df = filter_by_value(df, target_col, selected_value)
            
            # 6. Display the filtered results with highlighting
            def highlight_col(s):
                if s.name == target_col:
                    return ['background-color: #1f77b4; color: white'] * len(s)
                return [''] * len(s)
            
            st.dataframe(
                filtered_df.style.apply(highlight_col, axis=0),
                width='stretch',
                hide_index=True
            )
        else:
            # Default state when nothing is clicked
            st.info("Click a value in the left table to see the full records here.")

