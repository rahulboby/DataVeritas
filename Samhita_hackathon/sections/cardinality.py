import streamlit as st
import pandas as pd 
import plotly.express as px

def displayCardinalityStats(df):
    # --- 0. CUSTOM CSS ---
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


    # --- 1. Original Calculations (RETAINED) ---
    cardinality_data = []
    for col in df.columns:
        unique_count = df[col].nunique(dropna=False)
        total_count = df.shape[0]
        cardinality_data.append({
            "column": col,
            "unique_count": int(unique_count),
            "total_count": int(total_count),
            "cardinality_ratio": round(unique_count / total_count, 4) if total_count > 0 else 0
        })

    cardinality_df = pd.DataFrame(cardinality_data)
    
    # --- 2. Enhanced Metrics ---
    # Identify potential Primary Keys (Ratio close to 1)
    pk_candidates = cardinality_df[cardinality_df["cardinality_ratio"] >= 0.90].shape[0]
    # Identify Constant Columns (Only 1 unique value)
    constant_cols = cardinality_df[cardinality_df["unique_count"] <= 1].shape[0]
    # Average Ratio
    avg_ratio = cardinality_df["cardinality_ratio"].mean()

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Columns", len(df.columns))
        st.expander("Show").write(f"Average Cardinality Ratio: {avg_ratio:.2%}")
    with m_col2:
        st.metric("High Cardinal Columns", pk_candidates, help="Columns where >90% of values are unique (Potential Keys).")
        st.expander("Show").write(cardinality_df[cardinality_df["cardinality_ratio"] >= 0.90][["column", "cardinality_ratio"]].reset_index(drop=True))
    with m_col3:
        st.metric("Constant Columns", constant_cols, help="Columns with 1 or 0 unique values (No information gain).", delta_color="inverse")
        st.expander("Show").write(cardinality_df[cardinality_df["unique_count"] <= 1][["column", "unique_count"]].reset_index(drop=True))
    st.divider()

    # --- 3. Layout: Side-by-Side ---
    col1, col2 = st.columns([1, 1])

    # Sort by cardinality ratio descending (RETAINED)
    cardinality_df = cardinality_df.sort_values(by="cardinality_ratio", ascending=False)

    with col1:
        st.subheader("Raw Cardinality Data")
        # Beautified dataframe display
        st.dataframe(
            cardinality_df.style.background_gradient(subset=['cardinality_ratio'], cmap='Blues'),
            width = 'stretch',
            height=450
        )

    with col2:
        st.subheader("High Cardinality Distribution")
        # Filtered logic (RETAINED)
        top_x = 5 # How many top fields to show in the plot
        filtered_cardinality_df = cardinality_df.sort_values(by = "cardinality_ratio").head(top_x)
        
        if not filtered_cardinality_df.empty:
            fig_bar = px.bar(
                filtered_cardinality_df.sort_values(by = 'cardinality_ratio', ascending=True),
                x="cardinality_ratio",
                y="column",
                orientation="h",
                text="cardinality_ratio",
                color="cardinality_ratio",
                color_continuous_scale="Blues"
            )

            fig_bar.update_layout(
                xaxis_title="Ratio (Unique / Total)",
                yaxis_title="",
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False,
                height=450
            )
            fig_bar.update_xaxes(showgrid=True, gridcolor='#E2E8F0')
            fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            
            st.plotly_chart(fig_bar)
            st.caption(f"Showing top {top_x} Fields")
        else:
            st.info("No columns found with a cardinality ratio higher than 0.2.")
