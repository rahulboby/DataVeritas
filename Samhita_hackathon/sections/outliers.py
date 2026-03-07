import plotly.express as px
import streamlit as st
import pandas as pd
from sections import value_distribution as vdm
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from core.outliers.outlier_score import getOutlierScore

def displayOutlierStats(df: pd.DataFrame):

    
    # Get column stats
    _, _, _, _, numeric_cols, _, _ = vdm.get_column_stats(df)
    
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset.")
        return

    with st.expander("Show Outlier Records"):
        _, outlier_df = getOutlierScore(df)
        st.dataframe(outlier_df)

    st.subheader("Field-Wise Outlier Explorer")
    num_outlier_rows = outlier_df.shape[0]
    num_valid_rows = df.shape[0] - num_outlier_rows


    col_1, col_2, col_3, col_4 = st.columns(4)
    # Column 1 given below 
    col_3.metric("Outlier Rows", f"{num_outlier_rows:,}")
    col_4.metric("Valid Rows", f"{num_valid_rows:,}")


        
    outlier_stats = []
    col_bounds = {}

    # 1. Calculate stats and bounds
    for col in numeric_cols:
        c = df[col].dropna()
        if len(c) == 0: continue
        
        q1, q3 = c.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_limit = q1 - 5.0 * iqr
        upper_limit = q3 + 5.0 * iqr
        
        col_bounds[col] = (lower_limit, upper_limit)
        
        col_outliers = c[(c < lower_limit) | (c > upper_limit)]
        outlier_stats.append({
            "Column": col,
            "No. Outliers": len(col_outliers),
            "Outlier %": (len(col_outliers) / len(c)) * 100.0 if len(c) > 0 else 0
        })

    outlier_stats_df = pd.DataFrame(outlier_stats)
    outlier_stats_df = outlier_stats_df.sort_values(by="Outlier %", ascending=False).reset_index(drop=True)
    
    col_1.metric("Fields with Outliers", f"{len(outlier_stats_df[outlier_stats_df['No. Outliers']!=0]):,}")
    col_2.metric("Numeric Columns", f"{len(numeric_cols):,}")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Outliers in Fields")
        # Enable single-row selection
        selection = st.dataframe(
            outlier_stats_df[outlier_stats_df['No. Outliers']!=0].head(5000),
            on_select="rerun",
            selection_mode="single-row", 
            hide_index=True,
            width = 'stretch'
        )    
        st.caption("Click a row to visualize the distribution")
        with st.expander("Clean Columns (No Outliers Detected)"):
            st.dataframe(outlier_stats_df[outlier_stats_df['No. Outliers']==0].head(5000))


    with col2:
        st.subheader("Outlier Distribution of Selected Field")
        selected_col = None
        # 2. Extract selected column from dataframe selection
        # selection.selection['rows'] returns a list of indices
        selected_rows = selection.selection.get("rows", [])
        
        if selected_rows:
            # Get the column name from the selected row index
            row_idx = selected_rows[0]
            selected_col = outlier_stats_df.iloc[row_idx]['Column']
            
            # 3. Prepare Plotting Data
            plot_df = df[[selected_col]].dropna().copy()
            lower, upper = col_bounds[selected_col]
            
            plot_df['Status'] = plot_df[selected_col].apply(
                lambda x: 'Outlier' if (x < lower or x > upper) else 'Normal'
            )

            # 4. Generate the Chart
            fig = px.box(
                plot_df, 
                y=selected_col, 
                color="Status",
                color_discrete_map={'Normal': '#636EFA', 'Outlier': '#EF553B'},
                title=f"Distribution for: {selected_col}",
                template="plotly_white"
            )

            # Normal data: Show Box, Hide Points
            fig.update_traces(
                boxpoints=False, 
                selector=dict(name="Normal")
            )

            # Outlier data: Hide Box, Show Points
            fig.update_traces(
                boxpoints="all",
                fillcolor="rgba(0,0,0,0)",
                line_color="rgba(0,0,0,0)",
                marker=dict(opacity=0.8, size=8),
                selector=dict(name="Outlier")
            )

            fig.update_layout(
                yaxis_title="Values",
                xaxis_title="",
                showlegend=True
            )

            st.plotly_chart(fig)
            
            # Outlier Details
            outlier_count = len(plot_df[plot_df['Status'] == 'Outlier'])
            st.info(f"**{selected_col}** has **{outlier_count}** outliers outside the range: [{lower:,.2f} to {upper:,.2f}]")
        else:
            st.info("Select a row in the table to view the outlier spread.")
    if selected_col is not None:
        with st.expander(f"Show the records for outliers in {selected_col}"):
            lb, ub = col_bounds[selected_col]
            st.dataframe(
                df[(df[selected_col]>ub) | (df[selected_col]<lb)].head(5000)
            )
    st.divider()
    
    

