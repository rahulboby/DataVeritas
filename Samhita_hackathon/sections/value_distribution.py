import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import numpy as np
from core.value_distribution.columns_stats import get_column_stats

# --- MAIN DISPLAY ---
def displayValueDistributionStats(df):
    apply_custom_css()

    counts, constant_cols, datetime_cols, boolean_cols, numeric_cols, categorical_cols, identifier_cols = get_column_stats(df)
    filtered_counts = {k: v for k, v in counts.items() if v > 0}

    # --- TOP SECTION: PIE (LEFT) & QUARTILE TABLE (RIGHT) ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">Field Composition</div>', unsafe_allow_html=True)

        pie_df = pd.DataFrame({
            "Type": list(filtered_counts.keys()),
            "Count": list(filtered_counts.values())
        })

        fig_pie = px.pie(
            pie_df,
            names="Type",
            values="Count",
            hole=0.45
        )

        fig_pie.update_traces(
            textposition='outside', 
            textinfo='label+value'
        )
        fig_pie.update_layout(
            annotations=[dict(text=f"Total<br>{df.shape[1]:,}", x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=False, 
            margin=dict(l=40, r=40, t=20, b=40),
            height=380
        )
        
        
        
        # --- changed: use streamlit_plotly_events instead of st.plotly_chart(on_select=...) ---
        st.plotly_chart(fig_pie)
    summary_df = None    
    with col_right:
        st.markdown('<div class="section-header">Numeric Summary</div>', unsafe_allow_html=True)
        if numeric_cols:
            summary_data = []
            for col in numeric_cols:
                c = df[col].dropna()
                if not c.empty:
                    summary_data.append({
                        'Field': col, 'Mean': c.mean(), 'Min': c.min(), 
                        'Median': c.median(), 'Max': c.max()
                    })
            summary_df = pd.DataFrame(summary_data).set_index('Field')
            st.dataframe(
                summary_df.head(5000).style.background_gradient(cmap='Blues', subset=['Mean', 'Median']).format("{:,.2f}"),
                width = 'stretch', height=320
            )
        else:
            st.info("No numeric columns available.")
    


    # --- MIDDLE SECTION: CATEGORICAL DENSITY ---
    st.markdown('<div class="section-header">Distinct Values in Categorical Columns</div>', unsafe_allow_html=True)
    display_categorical_density_chart(df, categorical_cols, len(df))

    st.divider()

    # --- BOTTOM SECTION: EXPLORER ---
    with st.container(border=True):
        if numeric_cols:
            st.markdown('<div class="section-header">Distribution Explorer</div>', unsafe_allow_html=True)
            display_advanced_explorer(df, numeric_cols, summary_df)


def apply_custom_css():
    st.markdown("""
        <style>
        .main-title { font-size: 2.5rem !important; font-weight: 700; color: #1E3A8A; margin-bottom: 20px; }
        .section-header { font-size: 1.3rem; font-weight: 600; color: #334155; margin-top: 10px; margin-bottom: 15px; border-left: 5px solid #1E3A8A; padding-left: 15px; }
        </style>
    """, unsafe_allow_html=True)

# --- PICTORIAL COMPONENT: FREQUENCY-BASED GRADIENT STACKED BAR ---
def display_categorical_density_chart(df, categorical_cols, total_rows):
    if not categorical_cols:
        st.info("No categorical columns identified (using the <= 40 unique values rule).")
        return
    
    plot_data = []
    for col in categorical_cols:
        # 1. Get counts and calculate dominance %
        counts = df[col].value_counts().reset_index()
        counts.columns = ['Category', 'Occurrence']
        counts['Field'] = col
        counts['Dominance'] = (counts['Occurrence'] / total_rows) * 100
        
        # 2. Keep only Top 5
        counts = counts.head(5)
        
        # 3. Create a clean label for the inside of the bar
        # <br> creates a line break for better readability
        counts['Label'] = counts.apply(
            lambda x: f"<b>{x['Category']}</b><br>{x['Dominance']:.1f}%", axis=1
        )
        
        # 4. Sort descending so the largest (most dense) is at the bottom of the stack
        counts = counts.sort_values(by='Occurrence', ascending=False)
        plot_data.append(counts)

    combined_df = pd.concat(plot_data)

    # 5. Create Bar Chart
    fig = px.bar(
        combined_df,
        x="Field",
        y="Occurrence",
        color="Occurrence",
        text="Label", # Use our custom label
        hover_data=["Category", "Dominance"],
        color_continuous_scale='Blues',
        title="Top 5 frequency Distinct Values per Categorical Field"
    )

    fig.update_traces(
        texttemplate='%{text}', 
        textposition='inside',
        insidetextanchor='middle',
        marker=dict(line=dict(color='#ffffff', width=1)) # White border to separate stacks
    )
    
    fig.update_layout(
        coloraxis_showscale=False, # Hide scale as labels provide the info
        xaxis_title="Categorical Fields",
        yaxis_title="Record Count",
        barmode='stack',
        height=600,
        margin=dict(l=10, r=10, t=50, b=10),
        uniformtext_minsize=8, 
        uniformtext_mode='hide' # Hides text if the stack is too small to fit it
    )
    
    st.plotly_chart(fig)

    #Display all the categories in all categorical columns
    with st.expander("See Full Category Listings for All Categorical Fields"):
        # Let the user choose which columns to display
        selected_cols = st.multiselect("Select Columns to View:", df.columns)
        if not selected_cols:
            selected_cols = df.columns  # Default to all if none selected
        i = 0
        col1, col2 = st.columns(2)
        for col in selected_cols:
            if i%2==0:
                with col1:
                    st.markdown(f"**{col}** (Distinct Values: {df[col].nunique()})")
                    cat_values = pd.DataFrame(df[col].value_counts()).reset_index()
                    cat_values.columns = [col, "Occurrence Count"]
                    st.dataframe(cat_values)
            else:
                with col2:
                    st.markdown(f"**{col}** (Distinct Values: {df[col].nunique()})")
                    cat_values = pd.DataFrame(df[col].value_counts()).reset_index()
                    cat_values.columns = [col, "Occurrence Count"]
                    st.dataframe(cat_values)
            i += 1
import streamlit as st
import pandas as pd

# PART 1: Cached filtering logic (expensive operations cached here)
@st.cache_data
def filter_dataframe(df, col_name, mode, low_val, high_val, col_min, col_max):
    """
    Cached function that performs the actual filtering.
    Only recalculates if any parameter changes.
    """
    if mode == "Inside Range": 
        mask = df[col_name].between(low_val, high_val)
        str_to_display = f" is between **{low_val}** and **{high_val}**"
    elif mode == "Outside Range": 
        mask = (df[col_name] < low_val) | (df[col_name] > high_val)
        str_to_display = f" is outside **{low_val}** to **{high_val}**"
    elif mode == "Minimum Value": 
        str_to_display = f" is the minimum value of **{col_min}**"
        mask = (df[col_name] == col_min)
    elif mode == "Maximum Value": 
        str_to_display = f" is the maximum value of **{col_max}**"
        mask = (df[col_name] == col_max)
    
    return df[mask].head(1000), str_to_display


# PART 2: UI and interaction logic (with fragment to prevent full page reload)
@st.fragment
def display_advanced_explorer(df, numeric_cols, summary_df):
    # 1. Initialize session states
    if "adv_filtered_df" not in st.session_state: 
        st.session_state["adv_filtered_df"] = None
    if "adv_active_col" not in st.session_state: 
        st.session_state["adv_active_col"] = numeric_cols[0]
    if "adv_low_val" not in st.session_state:
        st.session_state["adv_low_val"] = 0.0
    if "adv_high_val" not in st.session_state:
        st.session_state["adv_high_val"] = 100.0
    if "adv_display_str" not in st.session_state:
        st.session_state["adv_display_str"] = ""

    # 2. Column Selection 
    sel_col = st.selectbox(
        "Explore Column:", 
        numeric_cols, 
        key="adv_col_select", 
        index=None, 
        placeholder="Select column to explore"
    )
    
    if sel_col is None:
        st.warning("Please select a numeric column to explore.")
        st.session_state["adv_filtered_df"] = None
        return

    # 3. Get column metadata (fast operations)
    is_float = pd.api.types.is_float_dtype(df[sel_col])
    col_min = float(summary_df.loc[sel_col, 'Min']) if is_float else int(summary_df.loc[sel_col, 'Min'])
    col_max = float(summary_df.loc[sel_col, 'Max']) if is_float else int(summary_df.loc[sel_col, 'Max'])
    
    # 4. Handle column switching
    if st.session_state["adv_active_col"] != sel_col:
        st.session_state["adv_active_col"] = sel_col
        st.session_state["adv_low_val"] = col_min
        st.session_state["adv_high_val"] = col_max
        st.session_state["adv_filtered_df"] = None
        st.session_state["adv_display_str"] = ""

    # 5. The Input Form
    with st.form("explorer_filter_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            sub_c1, sub_c2 = st.columns(2)
            input_step = 0.01 if is_float else 1
            
            with sub_c1:
                low_val_input = st.number_input(
                    "Lower Limit:", 
                    key="adv_low_val", 
                    step=input_step
                )
            with sub_c2:
                high_val_input = st.number_input(
                    "Upper Limit:", 
                    key="adv_high_val", 
                    step=input_step
                )
            
            st.caption(f"Actual Data Range: {col_min} to {col_max} ({'Float' if is_float else 'Integer'})")
            
        with c2:
            mode = st.radio(
                "Logic:", 
                ["Inside Range", "Outside Range", "Minimum Value", "Maximum Value"], 
                horizontal=True, 
                key="logic_radio"
            )

        submit_button = st.form_submit_button("Preview Table")

    # 6. Process on button click - using CACHED filtering function
    if submit_button:
        # This call is cached! Only recalculates if parameters change
        filtered_df, str_to_display = filter_dataframe(
            df, 
            sel_col, 
            mode, 
            low_val_input, 
            high_val_input, 
            col_min, 
            col_max
        )
        st.session_state["adv_filtered_df"] = filtered_df
        st.session_state["adv_display_str"] = str_to_display

    # 7. Display the Highlighted Table
    if st.session_state["adv_filtered_df"] is not None:
        res = st.session_state["adv_filtered_df"]
        st.write(f"Showing **{len(res):,}** records where **{sel_col}**" + st.session_state["adv_display_str"])

        # Inline styling function (faster than apply)
        def highlight_selected(s):
            if s.name == sel_col:
                return ['background-color: #2e7d32; color: white; font-weight: bold'] * len(s)
            return [''] * len(s)

        styled_df = res.style.apply(highlight_selected, axis=0)
        st.dataframe(styled_df)
    else:
        st.info("Adjust the limits and click **'Preview Table'** to load the data.")

