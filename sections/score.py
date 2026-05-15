import streamlit as st
from core.score.overall_field_score import getOverallFieldScore
from core.score.overall_score import getOverallScore
from core.consistency.consistency_score_and_df import getConsistencyScore
from core.outliers import outlier_score as OS

def displayScoreStats(df):
    st.markdown("""
        <style>
        .score-title {
            color: #13293d;
            font-size: 2rem;
            font-weight: 750;
            margin-bottom: 0.4rem;
        }
        .score-subtitle {
            color: #516477;
            margin-bottom: 1rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d7e1e8;
            border-radius: 12px;
            padding: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    # ---------------- SESSION STATE INIT ----------------
    if "dq_results" not in st.session_state:
        st.session_state.dq_results = getOverallScore(df)

    if "use_custom_weights" not in st.session_state:
        st.session_state.use_custom_weights = False

    # ---------------- CUSTOM WEIGHT CONTROLS ----------------
    st.checkbox(
        "Use custom weights for score components",
        key="use_custom_weights"
    )

    if st.session_state.use_custom_weights:
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            null_w = st.number_input(
                "Null Weight",
                min_value=0.0,
                value=0.1,
                step=0.05
            )

        with col2:
            completeness_w = st.number_input(
                "Completeness Weight",
                min_value=0.0,
                value=0.3,
                step=0.05
            )

        with col3:
            uniqueness_w = st.number_input(
                "Uniqueness Weight",
                min_value=0.0,
                value=0.3,
                step=0.05
            )

        with col4:
            outlier_w = st.number_input(
                "Outlier Weight",
                min_value=0.0,
                value=0.1,
                step=0.05
            )

        with col5:
            violation_w = st.number_input(
                "Violation Weight",
                min_value=0.0,
                value=0.2,
                step=0.05
            )

        with col6:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Compute Score", use_container_width=True):
                st.session_state.dq_results = getOverallScore(
                    df,
                    null_w=null_w,
                    completeness_w=completeness_w,
                    uniqueness_w=uniqueness_w,
                    outlier_w=outlier_w,
                    violation_w=violation_w
                )

        if st.button("Reset to Default Score"):
            st.session_state.dq_results = getOverallScore(df)

    # ---------------- USE PERSISTENT RESULTS ----------------
    dq_score, null_score, completeness_score, uniqueness_score, outlier_score, violation_score = st.session_state.dq_results

    # ---------------- MAIN SCORE SECTION ----------------
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Overall DQ Score", f"{dq_score:.1%}")
            st.progress(dq_score)

        with col2:
            st.write("### Data Health Status")

            if dq_score > 0.9:
                st.success("Trust Level: Excellent")
            elif dq_score > 0.7:
                st.warning("Trust Level: Moderate - Data requires minor cleaning.")
            elif dq_score > 0.5:
                st.warning("Trust Level: Low - Quality of Data")
            else:
                st.error("Critical: Data very unreliable.")

    # ---------------- COMPONENT BREAKDOWN ----------------
    st.subheader("Component Breakdown")
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric(
        "Non-Null Records Score",
        f"{null_score:.2%}",
        help="Ratio of records with complete data",
        delta="-Review" if (null_score - 0.90) < 0 else "Good"
    )

    m2.metric(
        "Completeness Score",
        f"{completeness_score:.2%}",
        help="Percentage of non-null values",
        delta="-Review" if (completeness_score - 0.90) < 0 else "Good"
    )

    m3.metric(
        "Uniqueness Score",
        f"{uniqueness_score:.2%}",
        help="Ratio of unique primary keys/records",
        delta="-Review" if (uniqueness_score - 0.90) < 0 else "Good"
    )

    m4.metric(
        "Outlier Score",
        f"{outlier_score:.2%}",
        help="Ratio of non-outlier (normal) records",
        delta="-Review" if (outlier_score - 0.90) < 0 else "Good"
    )

    m5.metric(
        "Data Consistency Score (Given Rules)",
        f"{violation_score:.2%}",
        help="Ratio of records that violate the given set of rules",
        delta="-Review" if (violation_score - 0.90) < 0 else "Good"
    )

    with m6:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.1f} MB")

    # ---------------- DATA EXTRACTION ----------------
    _, outlier_df = OS.getOutlierScore(df)
    _, violation_df = getConsistencyScore(df)

    with st.expander("Explore Data", expanded=False):
        selection = st.selectbox(
            "Select Record to show:",
            [
                'All Records',
                'Incomplete Records',
                'Complete (No Nulls) Records',
                'Data with Violation',
                'Data without violation',
                'Outlier Records (Numeric)',
                'Outlier-Free Records',
                'Unique Records'
            ],
            index=None,
            placeholder="Select record to show"
        )

        if selection == 'All Records':
            st.text("Showing all records")
            st.dataframe(df.head(50000))
            st.markdown(f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")

        elif selection == 'Incomplete Records':
            filtered = df[df.isnull().any(axis=1)]
            st.text("Showing records with at least one null value")
            st.dataframe(filtered.head(50000))
            st.markdown(f"{filtered.shape[0]:,} rows × {filtered.shape[1]:,} columns")

        elif selection == 'Complete (No Nulls) Records':
            filtered = df.dropna()
            st.text("Showing records with no null values")
            st.dataframe(filtered.head(50000))
            st.markdown(f"{filtered.shape[0]:,} rows × {filtered.shape[1]:,} columns")

        elif selection == 'Data with Violation':
            st.text("Showing records that violate the given set of rules")
            st.dataframe(violation_df.head(50000))
            st.markdown(f"{violation_df.shape[0]:,} rows × {violation_df.shape[1]:,} columns")

        elif selection == 'Data without violation':
            filtered = df[~df.index.isin(violation_df.index)]
            st.text("Showing records that do not violate the given set of rules")
            st.dataframe(filtered.head(50000))
            st.markdown(f"{filtered.shape[0]:,} rows × {filtered.shape[1]:,} columns")

        elif selection == 'Outlier Records (Numeric)':
            st.text("Showing records that are outliers in numeric columns")
            st.dataframe(outlier_df.head(50000))
            st.markdown(f"{outlier_df.shape[0]:,} rows × {outlier_df.shape[1]:,} columns")

        elif selection == 'Outlier-Free Records':
            filtered = df[~df.index.isin(outlier_df.index)]
            st.text("Showing records that are not outliers")
            st.dataframe(filtered.head(50000))
            st.markdown(f"{filtered.shape[0]:,} rows × {filtered.shape[1]:,} columns")

        elif selection == 'Unique Records':
            filtered = df.drop_duplicates()
            st.text("Showing non-duplicate records")
            st.dataframe(filtered.head(50000))
            st.markdown(f"{filtered.shape[0]:,} rows × {filtered.shape[1]:,} columns")

    # ---------------- FIELD SCORE ----------------
    st.markdown('<div class="score-title">Field-wise Data Trust Score</div>', unsafe_allow_html=True)

    with st.container(border=True):
        select_column = st.selectbox(
            "Select a column to view field-wise scores:",
            df.columns,
            placeholder="Select column",
            index=None
        )

        if select_column:
            overall_field_score, null_score_field, unique_score_field, outlier_score_field = getOverallFieldScore(df, select_column)

            st.subheader(f"Overall DQ Score for {select_column}: {overall_field_score:.1%}")

            with st.container(border=True):
                col_left, col_right = st.columns([1, 2])

                with col_left:
                    st.metric("Overall DQ Score", f"{overall_field_score:.1%}")
                    st.progress(overall_field_score)

                with col_right:
                    st.write("### Field Health Status")

                    if overall_field_score > 0.9:
                        st.success("Excellent")
                    elif overall_field_score > 0.7:
                        st.warning("Flagged: Data requires minor cleaning.")
                    else:
                        st.error("Critical: Data very unreliable.")

            st.subheader(f"Quality Component Scores for {select_column}")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Non-Null Score",
                    f"{null_score_field:.2%}",
                    delta="-Review" if (null_score_field - 0.90) < 0 else "Good"
                )

            with c2:
                st.metric(
                    "Unique Score",
                    f"{unique_score_field:.2%}",
                    delta="-Review" if (unique_score_field - 0.90) < 0 else "Good"
                )

            with c3:
                st.metric(
                    "Outlier Score",
                    f"{outlier_score_field:.2%}",
                    delta="-Review" if (outlier_score_field - 0.90) < 0 else "Good"
                )