import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def displayCorrelationTool(df: pd.DataFrame):
    apply_custom_css()

    numeric_cols = get_numeric_columns(df)

    if len(numeric_cols) < 2:
        st.info("At least two numeric columns are required to view correlations.")
        return

    controls_col1, controls_col2, controls_col3 = st.columns([2, 2, 1])

    with controls_col1:
        x_col = st.selectbox(
            "X Axis",
            numeric_cols,
            index=0,
            key="corr_x_col",
        )

    with controls_col2:
        y_col = st.selectbox(
            "Y Axis",
            numeric_cols,
            index=1,
            key="corr_y_col",
        )

    with controls_col3:
        max_points = st.number_input(
            "Max Points",
            min_value=500,
            max_value=50000,
            value=10000,
            step=500,
            key="corr_max_points",
            help="Limits rendered points for large datasets.",
        )

    if x_col == y_col:
        st.warning("Choose two different numeric columns.")
        return

    pair_df, metrics, line_df = calculate_correlation_view(df, x_col, y_col, int(max_points))

    if pair_df.empty or metrics["complete_pairs"] < 2:
        st.info("The selected columns do not have enough overlapping non-null values.")
        return

    render_metric_row(metrics)

    st.divider()

    fig = px.scatter(
        pair_df,
        x=x_col,
        y=y_col,
        opacity=0.65,
        color_discrete_sequence=["#1f5a7a"],
    )

    if line_df is not None and not line_df.empty:
        fig.add_scatter(
            x=line_df[x_col],
            y=line_df["Predicted"],
            mode="lines",
            name="Linear fit",
            line=dict(color="#c43d3d", width=3),
        )

    fig.update_layout(
        height=620,
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#E2E8F0")

    st.plotly_chart(fig)

    with st.expander("Preview Complete Pairs", expanded=False):
        st.dataframe(pair_df.head(10000), width="stretch", hide_index=True)


def apply_custom_css():
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number], exclude=["bool"]).columns.tolist()


@st.cache_data
def calculate_correlation_view(df: pd.DataFrame, x_col: str, y_col: str, max_points: int):
    complete_df = df[[x_col, y_col]].dropna().copy()
    complete_pairs = len(complete_df)
    missing_pair_rows = len(df) - complete_pairs

    if complete_pairs == 0:
        return complete_df, build_empty_metrics(missing_pair_rows), None

    metrics = {
        "complete_pairs": complete_pairs,
        "missing_pair_rows": missing_pair_rows,
        "pearson": complete_df[x_col].corr(complete_df[y_col], method="pearson"),
        "spearman": complete_df[x_col].corr(complete_df[y_col], method="spearman"),
        "covariance": complete_df[x_col].cov(complete_df[y_col]),
    }
    metrics["r_squared"] = metrics["pearson"] ** 2 if pd.notna(metrics["pearson"]) else np.nan

    line_df = build_regression_line(complete_df, x_col, y_col)

    if complete_pairs > max_points:
        plot_df = complete_df.sample(n=max_points, random_state=42).sort_values(x_col)
    else:
        plot_df = complete_df.sort_values(x_col)

    return plot_df, metrics, line_df


def build_empty_metrics(missing_pair_rows: int):
    return {
        "complete_pairs": 0,
        "missing_pair_rows": missing_pair_rows,
        "pearson": np.nan,
        "spearman": np.nan,
        "covariance": np.nan,
        "r_squared": np.nan,
    }


def build_regression_line(pair_df: pd.DataFrame, x_col: str, y_col: str):
    if pair_df[x_col].nunique() < 2:
        return None

    slope, intercept = np.polyfit(pair_df[x_col], pair_df[y_col], 1)
    x_values = np.linspace(pair_df[x_col].min(), pair_df[x_col].max(), 100)

    return pd.DataFrame(
        {
            x_col: x_values,
            "Predicted": slope * x_values + intercept,
        }
    )


def render_metric_row(metrics: dict):
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Complete Pairs", f"{metrics['complete_pairs']:,}")
    col2.metric("Pearson r", format_metric(metrics["pearson"]))
    col3.metric("Spearman rho", format_metric(metrics["spearman"]))
    col4.metric("R-squared", format_metric(metrics["r_squared"]))
    col5.metric("Covariance", format_metric(metrics["covariance"]))

    if metrics["missing_pair_rows"] > 0:
        st.caption(f"{metrics['missing_pair_rows']:,} rows were excluded because at least one selected value was null.")


def format_metric(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.4f}"
