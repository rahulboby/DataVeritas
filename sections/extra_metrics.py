import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.value_distribution.columns_stats import get_column_stats
from core.value_distribution.distribution_characteristics import (
    compute_categorical_distribution_characteristics,
    compute_numeric_distribution_characteristics,
)


def displayExtraMetricsStats(df):
    apply_custom_css()

    metrics = calculate_extra_metrics(df)
    numeric_df = metrics["numeric_df"]
    categorical_df = metrics["categorical_df"]
    relationship_df = metrics["relationship_df"]

    render_summary_metrics(df, numeric_df, categorical_df, relationship_df)
    st.divider()

    tab_numeric, tab_categorical, tab_relationships = st.tabs(
        ["Distribution Shape", "Class Balance & Entropy", "Coefficient Relationships"]
    )

    with tab_numeric:
        render_numeric_metrics(df, numeric_df, metrics["numeric_cols"])

    with tab_categorical:
        render_categorical_metrics(df, categorical_df)

    with tab_relationships:
        render_relationship_metrics(df, relationship_df, metrics["numeric_cols"])


def apply_custom_css():
    st.markdown(
        """
        <style>
        .section-header {
            font-size: 1.25rem;
            font-weight: 650;
            color: #334155;
            margin-top: 0.6rem;
            margin-bottom: 0.8rem;
            border-left: 5px solid #1E3A8A;
            padding-left: 14px;
        }
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


@st.cache_data
def calculate_extra_metrics(df):
    (
        _counts,
        constant_cols,
        _datetime_cols,
        boolean_cols,
        numeric_cols,
        categorical_cols,
        identifier_cols,
    ) = get_column_stats(df)

    class_metric_limit = max(2, min(40, int(max(1, df.shape[0] * 0.25))))
    low_cardinality_identifiers = [
        col for col in identifier_cols if df[col].nunique(dropna=True) <= class_metric_limit
    ]
    categorical_like_cols = list(
        dict.fromkeys(categorical_cols + boolean_cols + constant_cols + low_cardinality_identifiers)
    )

    numeric_characteristics = compute_numeric_distribution_characteristics(df, numeric_cols)
    categorical_characteristics = compute_categorical_distribution_characteristics(
        df,
        categorical_like_cols,
    )
    relationship_records = build_relationship_records(df, numeric_cols)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_like_cols,
        "numeric_df": build_numeric_metrics_df(numeric_characteristics),
        "categorical_df": build_categorical_metrics_df(categorical_characteristics),
        "relationship_df": pd.DataFrame(relationship_records),
    }


def build_numeric_metrics_df(characteristics):
    rows = []
    for field, values in characteristics.items():
        rows.append(
            {
                "Field": field,
                "Mean": values.get("mean", np.nan),
                "Median": values.get("median", np.nan),
                "Std Dev": values.get("std", np.nan),
                "Variance": values.get("variance", np.nan),
                "Coefficient of Variation": values.get("coefficient_of_variation", np.nan),
                "Skewness": values.get("skewness", np.nan),
                "Kurtosis": values.get("kurtosis", np.nan),
                "Q1": values.get("q1", np.nan),
                "Q3": values.get("q3", np.nan),
                "IQR": values.get("iqr", np.nan),
                "IQR Outliers": values.get("iqr_outlier_count", 0),
                "Feature Quality Score": values.get("feature_quality_score", 0),
                "Is Near Constant": values.get("is_near_constant", False),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Field",
                "Mean",
                "Median",
                "Std Dev",
                "Variance",
                "Coefficient of Variation",
                "Skewness",
                "Kurtosis",
                "Q1",
                "Q3",
                "IQR",
                "IQR Outliers",
                "Feature Quality Score",
                "Is Near Constant",
            ]
        )

    return pd.DataFrame(rows).sort_values("Feature Quality Score", ascending=True).reset_index(drop=True)


def build_categorical_metrics_df(characteristics):
    rows = []
    for field, values in characteristics.items():
        top_frequency = values.get("top_frequency", 0)
        rows.append(
            {
                "Field": field,
                "Top Value": values.get("top_value", ""),
                "Top Frequency": top_frequency,
                "Top Share %": values.get("top_share_pct", np.nan),
                "Class Count": values.get("num_classes", 0),
                "Imbalance Ratio": values.get("imbalance_ratio", np.nan),
                "Class Balance Score": values.get("class_balance_score", np.nan),
                "Entropy": values.get("entropy", np.nan),
                "Distribution Uniformity": values.get("distribution_uniformity", np.nan),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Field",
                "Top Value",
                "Top Frequency",
                "Top Share %",
                "Class Count",
                "Imbalance Ratio",
                "Class Balance Score",
                "Entropy",
                "Distribution Uniformity",
            ]
        )

    return pd.DataFrame(rows).sort_values("Class Balance Score", ascending=True).reset_index(drop=True)


def build_relationship_records(df, numeric_cols):
    if len(numeric_cols) < 2:
        return []

    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []

    corr_matrix = numeric_df.corr(method="pearson")
    records = []

    for left_idx in range(len(corr_matrix.columns)):
        for right_idx in range(left_idx + 1, len(corr_matrix.columns)):
            field_1 = corr_matrix.columns[left_idx]
            field_2 = corr_matrix.columns[right_idx]
            coefficient = corr_matrix.iloc[left_idx, right_idx]

            if pd.isna(coefficient):
                continue

            records.append(
                {
                    "Field 1": field_1,
                    "Field 2": field_2,
                    "Coefficient": round(float(coefficient), 4),
                    "Absolute Coefficient": round(abs(float(coefficient)), 4),
                    "Direction": "Positive" if coefficient >= 0 else "Negative",
                }
            )

    return sorted(records, key=lambda item: item["Absolute Coefficient"], reverse=True)


def render_summary_metrics(df, numeric_df, categorical_df, relationship_df):
    skewed_fields = int((numeric_df["Skewness"].abs() >= 1.0).sum()) if not numeric_df.empty else 0
    heavy_tail_fields = int((numeric_df["Kurtosis"] >= 3.0).sum()) if not numeric_df.empty else 0
    imbalanced_fields = (
        int((categorical_df["Class Balance Score"] < 50).sum()) if not categorical_df.empty else 0
    )
    high_relationships = (
        int((relationship_df["Absolute Coefficient"] >= 0.8).sum()) if not relationship_df.empty else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skewed Numeric Fields", skewed_fields, help="Absolute skewness at or above 1.0.")
    col2.metric("Heavy-Tailed Fields", heavy_tail_fields, help="Excess kurtosis at or above 3.0.")
    col3.metric("Imbalanced Class Fields", imbalanced_fields, help="Class balance score below 50.")
    col4.metric("Strong Relationships", high_relationships, help="Absolute Pearson coefficient at or above 0.80.")

    st.caption(
        f"Analyzing {len(df):,} rows across {len(numeric_df):,} numeric metric fields, "
        f"{len(categorical_df):,} categorical/binary fields, and {len(relationship_df):,} numeric field pairs."
    )


def render_numeric_metrics(df, numeric_df, numeric_cols):
    st.markdown('<div class="section-header">Numeric Distribution Metrics</div>', unsafe_allow_html=True)

    if numeric_df.empty:
        st.info("No numeric columns available for skewness, kurtosis, or coefficient metrics.")
        return

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        with c1:
            selected_fields = st.multiselect(
                "Fields",
                numeric_df["Field"].tolist(),
                default=numeric_df["Field"].tolist(),
                key="extra_numeric_fields",
            )

        with c2:
            metric_to_plot = st.selectbox(
                "Visualize",
                ["Skewness", "Kurtosis", "Coefficient of Variation", "Feature Quality Score"],
                key="extra_numeric_metric",
            )

        with c3:
            min_quality = st.slider(
                "Min Quality",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key="extra_numeric_quality",
            )

        with c4:
            flagged_only = st.checkbox(
                "Flagged only",
                value=False,
                key="extra_numeric_flagged_only",
                help="Show fields with skewness >= 1, kurtosis >= 3, high CV, or low feature quality.",
            )

    filtered_df = numeric_df[numeric_df["Field"].isin(selected_fields)].copy()
    filtered_df = filtered_df[filtered_df["Feature Quality Score"] >= min_quality]
    filtered_df["Metric Flags"] = filtered_df.apply(build_numeric_flags, axis=1)

    if flagged_only:
        filtered_df = filtered_df[filtered_df["Metric Flags"] != "OK"]

    if filtered_df.empty:
        st.info("No numeric fields match the selected filters.")
        return

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Metric Table")
        st.dataframe(
            filtered_df.style.background_gradient(
                cmap="Blues",
                subset=["Feature Quality Score"],
            ).format(
                {
                    "Mean": "{:,.4f}",
                    "Median": "{:,.4f}",
                    "Std Dev": "{:,.4f}",
                    "Variance": "{:,.4f}",
                    "Coefficient of Variation": "{:,.4f}",
                    "Skewness": "{:,.4f}",
                    "Kurtosis": "{:,.4f}",
                }
            ),
            width="stretch",
            height=430,
        )

    with col_right:
        st.subheader(f"{metric_to_plot} by Field")
        chart_df = filtered_df.sort_values(metric_to_plot, ascending=True).tail(20)
        fig = px.bar(
            chart_df,
            x=metric_to_plot,
            y="Field",
            orientation="h",
            color=metric_to_plot,
            color_continuous_scale="Blues",
            hover_data=["Metric Flags", "Feature Quality Score"],
        )
        fig.update_layout(
            height=430,
            xaxis_title=metric_to_plot,
            yaxis_title="",
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig)

    render_numeric_record_filter(df, numeric_cols)


def build_numeric_flags(row):
    flags = []

    if abs(row["Skewness"]) >= 1.0:
        flags.append("Skewed")
    if row["Kurtosis"] >= 3.0:
        flags.append("Heavy tail")
    if abs(row["Coefficient of Variation"]) >= 1.0:
        flags.append("High CV")
    if row["Feature Quality Score"] < 70:
        flags.append("Low quality")
    if bool(row["Is Near Constant"]):
        flags.append("Near constant")

    return ", ".join(flags) if flags else "OK"


@st.cache_data
def filter_numeric_records(df, column_name, mode, low_value, high_value):
    series = df[column_name]

    if mode == "Inside Range":
        mask = series.between(low_value, high_value)
        description = f"is between {low_value} and {high_value}"
    elif mode == "Outside Range":
        mask = (series < low_value) | (series > high_value)
        description = f"is outside {low_value} to {high_value}"
    elif mode == "Lower Tail":
        mask = series <= low_value
        description = f"is at or below {low_value}"
    else:
        mask = series >= high_value
        description = f"is at or above {high_value}"

    return df[mask].head(10000), description


@st.fragment
def render_numeric_record_filter(df, numeric_cols):
    with st.expander("Filter Source Records by Numeric Metric Field", expanded=False):
        selected_col = st.selectbox(
            "Metric field",
            numeric_cols,
            index=0 if numeric_cols else None,
            key="extra_numeric_filter_col",
        )

        if not selected_col:
            st.info("Select a numeric field to filter records.")
            return

        series = df[selected_col].dropna()
        if series.empty:
            st.info("The selected field has no non-null numeric values.")
            return

        is_float = pd.api.types.is_float_dtype(series)
        min_value = float(series.min())
        max_value = float(series.max())
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        step = 0.01 if is_float else 1.0

        with st.form("extra_numeric_record_filter_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                mode = st.radio(
                    "Logic",
                    ["Inside Range", "Outside Range", "Lower Tail", "Upper Tail"],
                    horizontal=True,
                    key="extra_numeric_filter_mode",
                )

            with c2:
                low_value = st.number_input(
                    "Lower value",
                    value=q1,
                    min_value=min_value,
                    max_value=max_value,
                    step=step,
                    key="extra_numeric_low_value",
                )

            with c3:
                high_value = st.number_input(
                    "Upper value",
                    value=q3,
                    min_value=min_value,
                    max_value=max_value,
                    step=step,
                    key="extra_numeric_high_value",
                )

            st.caption(f"Actual data range: {min_value:,.4f} to {max_value:,.4f}")
            submitted = st.form_submit_button("Preview Records")

        if submitted:
            filtered_df, description = filter_numeric_records(
                df,
                selected_col,
                mode,
                low_value,
                high_value,
            )
            st.session_state["extra_numeric_filtered_df"] = filtered_df
            st.session_state["extra_numeric_filter_desc"] = description
            st.session_state["extra_numeric_filter_field"] = selected_col

        if st.session_state.get("extra_numeric_filtered_df") is not None:
            result_df = st.session_state["extra_numeric_filtered_df"]
            desc = st.session_state["extra_numeric_filter_desc"]
            field = st.session_state["extra_numeric_filter_field"]
            st.write(f"Showing **{len(result_df):,}** records where **{field}** {desc}.")
            st.dataframe(result_df, width="stretch", hide_index=True)


def render_categorical_metrics(df, categorical_df):
    st.markdown('<div class="section-header">Class Balance, Imbalance, and Entropy</div>', unsafe_allow_html=True)

    if categorical_df.empty:
        st.info("No categorical, boolean, or constant columns available for class balance metrics.")
        return

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])

        with c1:
            selected_fields = st.multiselect(
                "Fields",
                categorical_df["Field"].tolist(),
                default=categorical_df["Field"].tolist(),
                key="extra_categorical_fields",
            )

        with c2:
            max_balance_score = st.slider(
                "Max Balance Score",
                min_value=0,
                max_value=100,
                value=100,
                step=5,
                key="extra_class_balance_filter",
                help="Lower scores indicate more concentrated or imbalanced values.",
            )

        with c3:
            min_top_share = st.slider(
                "Min Top Share",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key="extra_top_share_filter",
                help="Only show fields where the dominant class has at least this share.",
            )

    filtered_df = categorical_df[categorical_df["Field"].isin(selected_fields)].copy()
    filtered_df = filtered_df[
        (filtered_df["Class Balance Score"] <= max_balance_score)
        & (filtered_df["Top Share %"] >= min_top_share)
    ]

    if filtered_df.empty:
        st.info("No categorical fields match the selected filters.")
        return

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Balance Table")
        st.dataframe(
            filtered_df.style.background_gradient(
                cmap="Blues",
                subset=["Class Balance Score", "Distribution Uniformity"],
            ).format(
                {
                    "Top Share %": "{:,.2f}",
                    "Imbalance Ratio": "{:,.2f}",
                    "Class Balance Score": "{:,.1f}",
                    "Entropy": "{:,.4f}",
                    "Distribution Uniformity": "{:,.1f}",
                }
            ),
            width="stretch",
            height=430,
        )

    with col_right:
        st.subheader("Class Balance Score")
        chart_df = filtered_df.sort_values("Class Balance Score", ascending=True).head(20)
        fig = px.bar(
            chart_df,
            x="Class Balance Score",
            y="Field",
            orientation="h",
            color="Top Share %",
            color_continuous_scale="Blues",
            hover_data=["Top Value", "Top Share %", "Entropy", "Distribution Uniformity"],
        )
        fig.update_layout(
            height=430,
            xaxis_title="Balance Score",
            yaxis_title="",
            margin=dict(l=0, r=0, t=20, b=0),
            coloraxis_colorbar=dict(title="Top Share %"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(range=[0, 100])
        st.plotly_chart(fig)

    render_categorical_record_filter(df, filtered_df["Field"].tolist())


@st.cache_data
def get_categorical_value_counts(df, column_name):
    counts = df[column_name].value_counts(dropna=False).reset_index()
    counts.columns = ["Raw Value", "Count"]
    total = counts["Count"].sum()
    counts["Share %"] = counts["Count"] / total * 100 if total else 0
    counts["Value"] = counts["Raw Value"].map(format_filter_value)
    return counts[["Value", "Count", "Share %"]]


@st.cache_data
def filter_categorical_records(df, column_name, selected_values):
    display_series = df[column_name].map(format_filter_value)
    return df[display_series.isin(selected_values)].head(10000)


def format_filter_value(value):
    if pd.isna(value):
        return "<NULL>"
    return str(value)


@st.fragment
def render_categorical_record_filter(df, categorical_cols):
    with st.expander("Filter Source Records by Class Value", expanded=False):
        if not categorical_cols:
            st.info("No categorical fields match the active filters.")
            return

        selected_col = st.selectbox(
            "Class field",
            categorical_cols,
            key="extra_categorical_filter_col",
        )

        value_counts_df = get_categorical_value_counts(df, selected_col)
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.dataframe(
                value_counts_df.head(100).style.format({"Share %": "{:,.2f}"}),
                width="stretch",
                height=360,
                hide_index=True,
            )

        with col_right:
            selected_values = st.multiselect(
                "Values",
                value_counts_df["Value"].head(100).tolist(),
                key="extra_categorical_values",
            )

            if selected_values:
                filtered_df = filter_categorical_records(df, selected_col, selected_values)
                st.write(
                    f"Showing **{len(filtered_df):,}** records where **{selected_col}** "
                    f"is one of {len(selected_values):,} selected value(s)."
                )
                st.dataframe(filtered_df, width="stretch", hide_index=True)
            else:
                st.info("Select one or more values to preview matching records.")


def render_relationship_metrics(df, relationship_df, numeric_cols):
    st.markdown('<div class="section-header">Numeric Coefficient Relationships</div>', unsafe_allow_html=True)

    if len(numeric_cols) < 2 or relationship_df.empty:
        st.info("At least two numeric columns are required for coefficient relationships.")
        return

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])

        with c1:
            threshold = st.slider(
                "Minimum Absolute Coefficient",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="extra_relationship_threshold",
            )

        with c2:
            selected_fields = st.multiselect(
                "Fields in Matrix",
                numeric_cols,
                default=numeric_cols[: min(12, len(numeric_cols))],
                key="extra_relationship_fields",
            )

    filtered_relationships = relationship_df[
        (relationship_df["Absolute Coefficient"] >= threshold)
        & (relationship_df["Field 1"].isin(selected_fields))
        & (relationship_df["Field 2"].isin(selected_fields))
    ].copy()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Relationship Pairs")
        if filtered_relationships.empty:
            st.info("No field pairs match the selected coefficient threshold.")
        else:
            st.dataframe(
                filtered_relationships.head(5000).style.background_gradient(
                    cmap="Blues",
                    subset=["Absolute Coefficient"],
                ),
                width="stretch",
                height=430,
                hide_index=True,
            )

    with col_right:
        st.subheader("Correlation Matrix")
        matrix_cols = selected_fields[:]
        if len(matrix_cols) < 2:
            st.info("Select at least two numeric fields to render the matrix.")
        else:
            corr_matrix = df[matrix_cols].corr(method="pearson")
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
                text_auto=".2f",
            )
            fig.update_layout(
                height=430,
                margin=dict(l=0, r=0, t=20, b=0),
                coloraxis_colorbar=dict(title="Pearson"),
            )
            st.plotly_chart(fig)

    render_relationship_record_filter(df, relationship_df, threshold)


@st.fragment
def render_relationship_record_filter(df, relationship_df, threshold):
    with st.expander("Explore a Relationship Pair", expanded=False):
        filtered_relationships = relationship_df[relationship_df["Absolute Coefficient"] >= threshold]

        if filtered_relationships.empty:
            st.info("No relationship pairs are available at the active threshold.")
            return

        pair_labels = [
            f"{row['Field 1']} <> {row['Field 2']} ({row['Coefficient']:.4f})"
            for _, row in filtered_relationships.head(200).iterrows()
        ]

        selected_pair = st.selectbox(
            "Pair",
            pair_labels,
            index=None,
            placeholder="Select a pair to inspect",
            key="extra_relationship_pair",
        )

        if not selected_pair:
            st.info("Select a pair to render a scatter plot and record preview.")
            return

        selected_index = pair_labels.index(selected_pair)
        pair = filtered_relationships.iloc[selected_index]
        field_1 = pair["Field 1"]
        field_2 = pair["Field 2"]

        pair_df = df[[field_1, field_2]].dropna().copy()

        if pair_df.empty:
            st.info("The selected pair has no complete records.")
            return

        fig = px.scatter(
            pair_df.head(10000),
            x=field_1,
            y=field_2,
            opacity=0.65,
            title=f"{field_1} vs {field_2}",
        )
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=45, b=0))
        st.plotly_chart(fig)

        st.dataframe(df[[field_1, field_2]].dropna().head(10000), width="stretch", hide_index=True)
