import streamlit as st
@st.cache_resource(show_spinner="Finding exact duplicates...")
def getGlobalExactDuplicates(df, dup_count, dup_mask):
    if dup_count > 0:
        group_cols = list(df.columns)
        temp_dup = df[dup_mask].copy()
        temp_dup = temp_dup.reset_index() 
        
        grouped = temp_dup.groupby(group_cols, dropna=False).agg(
            occurrence_count=('index', 'size'),
            first_occurrence_index=('index', 'min')
        ).reset_index()
        
        grouped = grouped.rename(columns={'first_occurrence_index': 'First Occurrence'})
        
        # Reorder columns
        final_cols = ['occurrence_count', 'First Occurrence'] + group_cols
        
        # SORT AND RESET INDEX
        grouped = grouped[final_cols].sort_values(by='occurrence_count', ascending=False).reset_index(drop=True)

        # Professional Styling
        styled_df = grouped.head(1000).style.background_gradient(
            subset=['occurrence_count'], 
            cmap='Reds' 
        ).format(
            {'occurrence_count': "{:,}", 'First Occurrence': "{:,.0f}"}
        )
        return styled_df
    else:   
        return None