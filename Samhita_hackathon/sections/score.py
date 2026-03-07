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

    # st.markdown('<div class="score-subtitle">Composite score across completeness, uniqueness, outliers, and consistency.</div>', unsafe_allow_html=True)
    
    # 1. Get Scores
    dq_score, null_score, completeness_score, uniqueness_score, outlier_score, violation_score = getOverallScore(df)
    
    # 2. Main Hero Section (Big Score)
    # We use a container to group the main score
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            # Display score as a percentage
            st.metric("Overall DQ Score", f"{dq_score:.1%}")
            # Progress bar for visual impact
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


    # 3. Component Breakdown (Metrics in Columns)
    st.subheader("Component Breakdown")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    # Formatting helper: 0.95 -> 95%
    m1.metric("Non-Null Records Score", f"{null_score:.2%}", help="Ratio of records with complete data", delta= "-Review" if (null_score-0.90) < 0 else "Good")
    m2.metric("Completeness Score", f"{completeness_score:.2%}", help="Percentage of non-null values", delta= "-Review" if (completeness_score-0.90) < 0 else "Good")
    m3.metric("Uniqueness Score", f"{uniqueness_score:.2%}", help="Ratio of unique primary keys/records", delta= "-Review" if (uniqueness_score-0.90) < 0 else "Good")
    m4.metric("Outlier Score", f"{outlier_score:.2%}", help="Ratio of non-outlier (normal) records", delta= "-Review" if (outlier_score-0.90) < 0 else "Good")
    m5.metric("Data Consistency Score (Given Rules)", f"{violation_score:.2%}", help="Ratio of records that violate the given set of rules", delta= "-Review" if (violation_score-0.90) < 0 else "Good")
    with m6:
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory_mb:.1f} MB")
    
    # Get Violation and Outlier DF
    _, outlier_df = OS.getOutlierScore(df)
    _, violation_df = getConsistencyScore(df)

    with st.expander("Explore Data", expanded=False):
        selection = st.selectbox("Select Record to show: ", ['All Records', 'Incomplete Records','Complete (No Nulls) Records', 'Data with Violation', 'Data without violation', 'Outlier Records (Numeric)', 'Outlier-Free Records', 'Unique Records'], index=None, placeholder="Select record to show")

        if selection == 'All Records':
            st.text("Showing all records")
            st.dataframe(df.head(50000))
            st.markdown(f"{df.shape[0]:,} rows × {df.shape[1]:,} columns")
        elif selection == 'Incomplete Records':
            st.text("Showing records with at least one null value")
            st.dataframe(df[df.isnull().any(axis=1)].head(50000))
            st.markdown(f"{df[df.isnull().any(axis=1)].shape[0]:,} rows × {df[df.isnull().any(axis=1)].shape[1]:,} columns")
        elif selection == 'Complete (No Nulls) Records':
            st.text("Showing records with no null values")
            st.dataframe(df.dropna().head(50000))
            st.markdown(f"{df.dropna().shape[0]:,} rows × {df.dropna().shape[1]:,} columns")
            # The ',' format specifier in the formatted string adds comas as thousand separators to the number, making it easier to read when dealing with large datasets. For example, if df.shape[0] is 1000000, it will be displayed as 1,000,000.
        elif selection == 'Data with Violation':
            st.text("Showing records that violate the given set of rules")
            st.dataframe(violation_df.head(50000))
            st.markdown(f"{violation_df.shape[0]:,} rows × {violation_df.shape[1]:,} columns")
        elif selection == 'Data without violation':
            st.text("Showing records that do not violate the given set of rules")
            violation_free_df = df[~df.index.isin(violation_df.index)]
            st.dataframe(violation_free_df.head(50000))
            st.markdown(f"{violation_free_df.shape[0]:,} rows × {violation_free_df.shape[1]:,} columns")
        elif selection == 'Outlier Records (Numeric)':
            st.text("Showing records that are outliers in numeric columns (Isolation Forest)")
            st.dataframe(outlier_df.head(50000))
            st.markdown(f"{outlier_df.shape[0]:,} rows × {outlier_df.shape[1]:,} columns")
        elif selection == 'Outlier-Free Records':
            st.text("Showing records that are not outliers in numeric columns (Isolation Forest)")
            outlier_free_df = df[~df.index.isin(outlier_df.index)]
            st.dataframe(outlier_free_df.head(50000))
            st.markdown(f"{outlier_free_df.shape[0]:,} rows × {outlier_free_df.shape[1]:,} columns")
        elif selection == 'Unique Records':
            st.text("Showing non-duplicate records")
            st.dataframe(df.drop_duplicates().head(50000))
            st.markdown(f"{df.drop_duplicates().shape[0]:,} rows × {df.drop_duplicates().shape[1]:,} columns")
        
    st.markdown('<div class="score-title">Field-wise Data Trust Score</div>', unsafe_allow_html=True)
    with st.container(border=True):
        select_column = st.selectbox("Select a column to view field-wise scores:", df.columns, placeholder="Select column", index = None)

        if select_column:
            overall_field_score, null_score_field, unique_score_field, outlier_score_field = getOverallFieldScore(df, select_column)

            # FIELD SCORE
            st.subheader(f"Overall DQ Score for {select_column}: {overall_field_score:.1%}")
            with st.container(border=True):
                col_left, col_right = st.columns([1, 2])
                with col_left:
                    # Display score as a percentage
                    st.metric("Overall DQ Score", f"{overall_field_score:.1%}")
                    # Progress bar for visual impact
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
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Non - Null Score (Completeness)", f"{null_score_field:.2%}", help="Percentage of null values in this field", delta= "-Review" if (null_score_field-0.90) < 0 else "Good")
            with col2:
                st.metric("Unique Score", f"{unique_score_field:.2%}", help="Ratio of unique values in this field", delta= "-Review" if (unique_score_field-0.90) < 0 else "Good")
            with col3:
                st.metric("Outlier Score", f"{outlier_score_field:.2%}", help="Ratio of non-outlier (normal) records in this field", delta= "-Review" if (outlier_score_field-0.90) < 0 else "Good")




            
