
import streamlit as st
import pandas as pd

from core.duplicates.user_defined_merge_data import display_merge_data
from sections import value_distribution as vdm
from sections import duplicates as dm
from sections import nulls as nm
from sections import cardinality as cm
from sections import outliers as om
from sections import score as ScM
from sections import consistency as ConM
from core.data import generator as DG

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Data Trust Platform",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
        :root {
            --bg-top: #edf4f8;
            --bg-bottom: #f7fafc;
            --surface: #ffffff;
            --surface-alt: #f6f9fb;
            --border: #d7e1e8;
            --text-strong: #13293d;
            --text-muted: #516477;
            --brand: #1f5a7a;
            --brand-soft: #2d7aa0;
            --accent: #0f8b8d;
            --danger: #c43d3d;
        }

        .stApp {
            background:
                radial-gradient(1200px 500px at 0% -10%, #e7f1f7 0%, transparent 60%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
            color: var(--text-strong);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1200px 500px at 0% -10%, #e7f1f7 0%, transparent 60%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
        }

        html, body, [class*="css"] {
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }

        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2f45 0%, #174966 100%);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
        }

        [data-testid="stSidebar"] * {
            color: #eef5fb !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(234, 243, 251, 0.2);
        }

        [data-testid="stSidebar"] .stButton > button {
            border: 1px solid rgba(222, 236, 247, 0.28);
            background: rgba(255, 255, 255, 0.06);
            color: #f5fbff;
            border-radius: 10px;
            font-weight: 600;
            letter-spacing: 0.1px;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: rgba(255, 255, 255, 0.55);
            background: rgba(255, 255, 255, 0.12);
        }

        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"] > div {
            background: #ffffff;
            border-color: var(--border);
            color: var(--text-strong);
        }

        [data-baseweb="tab-list"] button {
            background: #edf4f8;
            color: var(--text-strong);
            border-radius: 8px;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        [data-testid="stDataFrame"] [role="columnheader"] {
            background: #edf4f8;
            color: var(--text-strong);
        }

        [data-testid="stDataFrame"] [role="gridcell"] {
            background: #ffffff;
            color: var(--text-strong);
        }

        .hero-shell {
            border: 1px solid var(--border);
            background: linear-gradient(135deg, #ffffff 0%, #f2f8fc 60%, #eef4f8 100%);
            border-radius: 16px;
            padding: 2.3rem 2rem;
            box-shadow: 0 10px 30px rgba(20, 52, 75, 0.07);
            margin-bottom: 1.4rem;
        }

        .hero-kicker {
            display: inline-block;
            font-size: 0.78rem;
            letter-spacing: 0.55px;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--brand);
            background: #eaf3f9;
            border: 1px solid #cfe2ee;
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            margin-bottom: 0.9rem;
        }

        .hero-title {
            font-size: 2.5rem;
            line-height: 1.18;
            margin: 0;
            color: var(--text-strong);
            font-weight: 750;
        }

        .hero-subtitle {
            margin-top: 0.8rem;
            color: var(--text-muted);
            font-size: 1.08rem;
            max-width: 760px;
        }

        .feature-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.5rem 1.2rem;
            min-height: 180px;
            box-shadow: 0 4px 12px rgba(20, 52, 75, 0.05);
        }

        .feature-title {
            color: var(--text-strong);
            margin: 0;
            font-size: 1.02rem;
            font-weight: 700;
        }

        .feature-copy {
            color: var(--text-muted);
            margin-top: 0.6rem;
            line-height: 1.45;
            font-size: 0.95rem;
        }

        .feature-tag {
            color: var(--brand-soft);
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .data-source-card {
            background: var(--surface);
            padding: 1.4rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(20, 52, 75, 0.07);
            max-width: 930px;
            margin: 1rem auto;
        }

        .section-heading {
            color: var(--text-strong);
            font-size: 1.6rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .section-subheading {
            color: var(--text-muted);
            margin-top: 0.35rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 12px;
            border-radius: 12px;
        }

        .stAlert {
            border-radius: 10px;
            border: 1px solid var(--border);
        }

        .dataframe {
            font-size: 0.9rem;
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'dataset_ready' not in st.session_state:
    st.session_state.dataset_ready = False
if 'dataset_label' not in st.session_state:
    st.session_state.dataset_label = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Data Preview"
if 'selected_cols' not in st.session_state:
    st.session_state.selected_cols = []

# ==================== CACHED LOADERS ====================
@st.cache_data(show_spinner="Loading CSV file...")
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data(show_spinner="Loading Excel file...")
def load_excel(file) -> pd.DataFrame:
    return pd.read_excel(file)

@st.cache_data(show_spinner="Generating synthetic data...")
def generate_data(n_rows: int) -> pd.DataFrame:
    return DG.get_data(n_rows)

# ==================== SIDEBAR ====================

# Sidebar used for navigation
with st.sidebar:
    st.markdown("## DataVeritas")
    st.caption("Enterprise Data Integrity & Trust Platform")
    st.divider()

    # Navigation (only show when dataset is ready)
    if st.session_state.dataset_ready and st.session_state.df is not None:
        
        # Render Preview Button
        if st.button(
                "Data Preview",
                width='stretch',
                type="primary" if st.session_state.current_page == "Data Preview" else "secondary"
            ):
                st.session_state.current_page = "Data Preview"
                st.rerun()
        
        # AUDIT BUTTONS
        pages = {
            "Trust Score": "Data Trust Score",
            "Distribution": "Value Distribution Audit",
            "Cardinality": "Cardinality Audit",
            "Duplicates": "Data Duplicates Audit",
            "Nulls": "Data Completeness Audit",
            "Outliers": "Anomaly Audit",
            "AI Readiness Score": "AI Readiness Score"
        }
        st.subheader("Audits")
        # Render the audit buttons
        for label, page_id in pages.items():
            if st.button(
                label,
                width='stretch',
                type="primary" if st.session_state.current_page == page_id else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()
        
        # TOOL BUTTONS
        tool_pages = {
            "Consistency": "Data Consistency Rule Engine (BETA)",
            "Data Merger": "Entity Resolution Engine (BETA)",
            "Data Drift Detector": "Dataset Drift Detector (BETA)"
        }
        st.subheader("Tools")
        # Render the rule buttons
        for label, page_id in tool_pages.items():
            if st.button(
                label,
                width='stretch',
                type="primary" if st.session_state.current_page == page_id else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()

        # # Show metrics of data - the shape of dataset
        # st.divider()

        # # Dataset info
        # st.markdown("### Current Dataset")
        # st.caption(f"**{st.session_state.dataset_label}**")

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.metric("Rows", f"{len(st.session_state.df):,}")
        # with col2:
        #     st.metric("Cols", len(st.session_state.df.columns))

        st.divider()

        # Clear dataset
        if st.button("Clear Dataset", width='stretch', type="secondary"):
            st.session_state.df = None
            st.session_state.dataset_ready = False
            st.session_state.dataset_label = ""
            st.session_state.current_page = "Data Preview"
            st.session_state.selected_cols = []
            st.rerun()
    else:
        st.info("Load a dataset to begin analysis.")

# ==================== MAIN CONTENT ====================

# If no dataset, show welcome screen
if not st.session_state.dataset_ready or st.session_state.df is None:
    # Welcome header
    st.markdown("""
        <div class='hero-shell'>
            <span class='hero-kicker'>AI Data Integrity & Trust Platform</span>
            <h1 class='hero-title'>Assess Data Reliability Before It Powers Decisions</h1>
            <p class='hero-subtitle'>
                Evaluate dataset integrity using trust scores, anomaly detection, rule validation, and data profiling from a single governance interface.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Feature showcase
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-tag'>Profiling</div>
                <h3 class='feature-title'>Comprehensive Data Scans</h3>
                <p class='feature-copy'>Evaluate core quality dimensions across high-volume datasets with fast previews and metrics.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-tag'>Scoring</div>
                <h3 class='feature-title'>Executive Quality KPIs</h3>
                <p class='feature-copy'>Track health indicators and expose risk areas that impact downstream reporting and automation.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-tag'>Resolution</div>
                <h3 class='feature-title'>Guided Record Merging</h3>
                <p class='feature-copy'>Resolve duplicates with weighted matching logic and controlled review workflows.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data source section at bottom
    st.markdown("""
        <div style='text-align: center; margin-top: 1rem; margin-bottom: 0.4rem;'>
            <h2 class='section-heading'>Load Data Source</h2>
            <p class='section-subheading'>Upload a production dataset or generate synthetic records for validation.</p>
        </div>
    """, unsafe_allow_html=True)

    # Data loading card
    with st.container():
        st.markdown("<div class='data-source-card'>", unsafe_allow_html=True)

        # Source selection
        tab1, tab2 = st.tabs(["Upload File", "Generate Sample Data (Testing Feature)"])

        with tab1:
            st.markdown("### Upload CSV or Excel File")
            st.caption("Supported formats: .csv, .xlsx, .xls (max 200MB)")

            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["csv", "xlsx", "xls"],
                help="Upload your dataset for quality analysis",
                label_visibility="collapsed"
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                load_file_button = st.button(
                    "Load File",
                    type="primary",
                    width='stretch',
                    disabled=uploaded_file is None
                )

            if load_file_button and uploaded_file is not None:
                try:
                    with st.spinner("Loading your data..."):
                        name = uploaded_file.name.lower()
                        if name.endswith(".csv"):
                            df = load_csv(uploaded_file)
                        else:
                            df = load_excel(uploaded_file)

                        st.session_state.df = df
                        st.session_state.dataset_ready = True
                        st.session_state.dataset_label = uploaded_file.name
                        st.session_state.selected_cols = df.columns.tolist()
                        st.success(f"Loaded: {uploaded_file.name}")
                        st.balloons()
                        st.rerun()

                except Exception as e:
                    st.session_state.dataset_ready = False
                    st.session_state.df = None
                    st.error(f"Error loading file: {str(e)}")

        with tab2:
            st.warning("DEPRECATED FEATURE: This feature is only used to test the application and will be removed soon.")
            st.markdown("### Generate Synthetic Dataset")
            st.caption("Create sample data for testing and exploration")

            col1, col2 = st.columns(2)

            with col1:
                n_rows = st.number_input(
                    "Number of rows",
                    min_value=100,
                    max_value=200000,
                    value=20000,
                    step=1000,
                    help="Specify how many rows to generate"
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Estimated Size", f"~{(n_rows * 0.001765):.1f} MB")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                generate_button = st.button(
                    "Generate Dataset",
                    type="primary",
                    width='stretch'
                )

            if generate_button:
                try:
                    with st.spinner(f"Generating {n_rows:,} rows of synthetic data..."):
                        df = generate_data(int(n_rows))

                        st.session_state.df = df
                        st.session_state.dataset_ready = True
                        st.session_state.dataset_label = f"Generated Dataset ({int(n_rows):,} rows)"
                        st.session_state.selected_cols = df.columns.tolist()
                        st.success(f"Generated {int(n_rows):,} rows")
                        st.balloons()
                        st.rerun()

                except Exception as e:
                    st.session_state.dataset_ready = False
                    st.session_state.df = None
                    st.error(f"Error generating data: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# ==================== PAGE ROUTING ====================
df = st.session_state.df
filtered_df = df[st.session_state.selected_cols] if st.session_state.selected_cols else df

current_page = st.session_state.current_page

# Page header
st.markdown(f"<h1 style='margin-bottom: 0;'>{current_page}</h1>", unsafe_allow_html=True)
st.caption(f"Dataset: {st.session_state.dataset_label} | {len(df):,} rows x {len(df.columns)} columns")
st.divider()

# Route to appropriate page
if current_page == "Data Preview":
    # Column filter section
    with st.expander("Column Filter", expanded=False):
        filter_col1, filter_col2 = st.columns([3, 1])

        with filter_col1:
            selected_cols = st.multiselect(
                "Select columns to display:",
                df.columns.tolist(),
                default=st.session_state.selected_cols,
                help="Choose which columns to display in the preview and analyze"
            )

        with filter_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Reset to All", width='stretch'):
                selected_cols = df.columns.tolist()
                st.rerun()

        if selected_cols:
            st.session_state.selected_cols = selected_cols
            filtered_df = df[selected_cols]
        else:
            st.warning("Please select at least one column.")
            st.session_state.selected_cols = df.columns.tolist()
            filtered_df = df

    # Data preview
    st.markdown("### Data Preview")
    st.caption(f"Showing first 10,000 rows of {len(df):,}")

    st.dataframe(
        filtered_df.head(10000),
        width='stretch',
        height=600
    )

    # Metrics
    st.divider()
    st.markdown("### Dataset Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        st.metric("Displayed Columns", len(filtered_df.columns))

    with col4:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")

elif current_page == "Data Trust Score":
    ScM.displayScoreStats(filtered_df)

elif current_page == "Value Distribution Audit":
    vdm.displayValueDistributionStats(filtered_df)

elif current_page == "Cardinality Audit":
    cm.displayCardinalityStats(filtered_df)

elif current_page == "Data Duplicates Audit":
    dm.displayDuplicateStats(filtered_df)

elif current_page == "Data Completeness Audit":
    nm.displayNullStats(filtered_df)

elif current_page == "Anomaly Audit":
    om.displayOutlierStats(filtered_df)

# The Tools: 
elif current_page == "Data Consistency Rule Engine (BETA)":
    st.warning("BETA Feature: Inaccurate results. Sill in processing.")
    ConM.displayConsistencyStats(filtered_df)

elif current_page == "Entity Resolution Engine (BETA)":
    st.warning("BETA feature: Data Merger uses weighted scoring. This feature performs best with datasets under 5,000 rows.")
    display_merge_data(filtered_df)

elif current_page == "Dataset Drift Detector (BETA)":
    st.warning("Still under Development")


# ==================== FOOTER ====================
st.divider()
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        <p style='margin: 0;'>DataVeritas v1.0 | Built with Streamlit | 2024</p>
    </div>
""", unsafe_allow_html=True)
