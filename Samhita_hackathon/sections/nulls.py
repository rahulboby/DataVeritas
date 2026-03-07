import streamlit as st
import pandas as pd
import plotly.express as px
from core.nulls.column_null_data import getColumnsNullData


# ------------------------ CACHED CALCULATIONS ------------------------

@st.cache_data
def calculate_null_statistics(df):
    """
    Calculate all null-related statistics once and cache the results.
    Returns a dictionary of all metrics.
    """
    nrows, ncols = df.shape
    total_cells = nrows * ncols
    
    # Basic null counts
    total_nulls = df.isnull().sum().sum()
    null_counts_per_col = df.isnull().sum()
    
    # Row-level analysis
    rows_with_nulls = df.isnull().any(axis=1).sum()
    clean_rows = nrows - rows_with_nulls
    
    # Overall metrics
    fill_rate = ((total_cells - total_nulls) / total_cells) * 100 if total_cells > 0 else 0
    
    # Worst column analysis
    if null_counts_per_col.max() > 0:
        worst_col_name = null_counts_per_col.idxmax()
        worst_col_count = int(null_counts_per_col.max())
        worst_col_pct = (worst_col_count / nrows) * 100
    else:
        worst_col_name = "None"
        worst_col_count = 0
        worst_col_pct = 0
    
    # Critical columns (>20% null)
    critical_threshold_pct = 20.0
    critical_threshold = (critical_threshold_pct / 100.0) * nrows
    critical_columns = [col for col in df.columns if null_counts_per_col[col] > critical_threshold]
    
    return {
        "nrows": nrows,
        "ncols": ncols,
        "total_cells": total_cells,
        "total_nulls": total_nulls,
        "fill_rate": fill_rate,
        "rows_with_nulls": rows_with_nulls,
        "clean_rows": clean_rows,
        "worst_col_name": worst_col_name,
        "worst_col_count": worst_col_count,
        "worst_col_pct": worst_col_pct,
        "critical_columns": critical_columns,
        "critical_threshold_pct": critical_threshold_pct,
        "null_counts_per_col": null_counts_per_col
    }


@st.cache_data
def prepare_chart_data(df, null_counts_per_col):
    """
    Prepare data for visualizations.
    Returns a dataframe ready for plotting.
    """
    # Filter only columns with nulls and sort
    null_counts = null_counts_per_col[null_counts_per_col > 0].sort_values(ascending=True)
    
    if null_counts.empty:
        return None
    
    nrows = len(df)
    chart_rows = []
    
    for col, nnull in null_counts.items():
        # Get unique values (limited to first 50 for performance)
        uniques = df[col].dropna().unique()[:50]
        uniques_str = ", ".join(map(str, uniques))
        if len(df[col].dropna().unique()) > 50:
            uniques_str += "..."
        
        chart_rows.append({
            "column": col,
            "nulls": int(nnull),
            "non_nulls": nrows - int(nnull),
            "total_rows": nrows,
            "unique_values": uniques_str
        })
    
    return pd.DataFrame(chart_rows)


# ------------------------ VISUALIZATION COMPONENTS ------------------------

def render_column_health_chart(chart_df):
    """Render the stacked horizontal bar chart for column health."""
    stacked_long = chart_df.melt(
        id_vars=["column"], 
        value_vars=["nulls", "non_nulls"], 
        var_name="status", 
        value_name="count"
    )
    stacked_long["status"] = stacked_long["status"].map({
        "nulls": "Missing", 
        "non_nulls": "Present"
    })

    fig = px.bar(
        stacked_long,
        x="count",
        y="column",
        color="status",
        orientation="h",
        barmode="stack",
        color_discrete_map={"Missing": "#C0392B", "Present": "#5185D4"},
        height=max(400, 35 * len(chart_df))
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Record Count",
        yaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#E2E8F0')
    
    return fig


def render_row_integrity_chart(stats):
    """Render the donut chart for row integrity."""
    fig = px.pie(
        values=[stats["clean_rows"], stats["rows_with_nulls"]],
        names=["Clean Records", "Incomplete Records"],
        hole=0.6,
        color=["Clean Records", "Incomplete Records"],
        color_discrete_map={
            "Incomplete Records": "#C0392B",
            "Clean Records": "#5185D4"
        }
    )
    
    fig.update_traces(
        textinfo="percent+label",
        insidetextorientation="horizontal",
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=40, b=0, l=0, r=0),
        height=400,
        annotations=[dict(
            text=f"Total<br>{stats['nrows']:,}",
            x=0.5,
            y=0.5,
            font_size=20,
            showarrow=False
        )]
    )
    
    return fig


# ------------------------ DRILL-DOWN SECTION (FRAGMENT) ------------------------

@st.fragment
def render_data_preview(df):
    """Fragment for the drill-down data preview section."""
    with st.expander("Data Preview", expanded=False):
        d_col1, d_col2 = st.columns([2, 1])
        
        with d_col1:
            selected_cols = st.multiselect(
                "Select columns to investigate:",
                df.columns.tolist(),
                key="null_drill_cols"
            )
        
        with d_col2:
            logic_mode = st.radio(
                "Logic:",
                options=["AND", "OR"],
                horizontal=True,
                key="null_drill_logic",
                help="AND: All selected must be null. OR: Any can be null."
            )

        if selected_cols:
            # This function is called only when user has selected columns
            match_count, masked_df = getColumnsNullData(df, selected_cols, logic_mode)
            st.write(f"Found **{match_count:,}** matching rows.")
            
            preview_button = st.button(
                "Generate Preview Table",
                type="primary",
                key="null_preview_btn"
            )
            
            if preview_button:
                st.session_state["null_show_table"] = True
            
            if st.session_state.get("null_show_table", False):
                st.dataframe(masked_df.head(50000), use_container_width=True)
        else:
            st.info("Select columns above to start filtering data.")


# ------------------------ MAIN DISPLAY FUNCTION ------------------------

def displayNullStats(df):
    """
    Main function to display null value analytics.
    Optimized with caching and fragments for performance.
    """
    
    # --- CUSTOM CSS ---
    st.markdown("""
        <style>
        .main-title {
            font-size: 36px;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 20px;
        }
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #C0392B !important;
        }
        </style>
    """, unsafe_allow_html=True)

    
    # --- CALCULATE STATISTICS (CACHED) ---
    stats = calculate_null_statistics(df)
    
    # --- HEADER METRICS ---
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric(
            "Data Completeness",
            f"{stats['fill_rate']:.2f}%",
            help="Percentage of non-null cells."
        )
    
    with m_col2:
        st.metric(
            f"Worst Column: {stats['worst_col_name']}",
            f"{stats['worst_col_pct']:.1f}% Null",
            f"{stats['worst_col_count']} rows",
            delta_color="inverse",
            help="Column with highest null values."
        )
    
    with m_col3:
        critical_str = ", ".join(stats['critical_columns']) if stats['critical_columns'] else "None"
        st.metric(
            label="Critical Column(s)",
            value=len(stats['critical_columns']),
            delta=critical_str if stats['critical_columns'] else None,
            delta_color="inverse",
            help=f"Columns with more than {stats['critical_threshold_pct']}% null values: {critical_str}"
        )

    st.divider()

    # --- PREPARE CHART DATA (CACHED) ---
    chart_df = prepare_chart_data(df, stats['null_counts_per_col'])

    # Early exit if no nulls
    if chart_df is None:
        st.success("Your dataset is 100% clean. No null values found.")
        return

    # --- VISUALIZATIONS ---
    col1, col2 = st.columns([2, 3])

    with col2:
        st.subheader("Column Health")
        fig_health = render_column_health_chart(chart_df)
        st.plotly_chart(fig_health, use_container_width=True)

    with col1:
        st.subheader(
            "Row Integrity",
            help="If at least one field in a record is null, it is marked as incomplete."
        )
        fig_integrity = render_row_integrity_chart(stats)
        st.plotly_chart(fig_integrity, use_container_width=True)

    # --- DRILL DOWN (FRAGMENT - ONLY THIS RERUNS) ---
    render_data_preview(df)

