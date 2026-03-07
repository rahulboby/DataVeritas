import streamlit as st
from io import BytesIO

def add_download_buttons(df, filename_prefix="data", show_header=True, show_data = False):
    """
    Add CSV and Excel download buttons for any dataframe.
    
    Args:
        your_df: The dataframe to download
        filename_prefix: Prefix for the downloaded filename (default: "data")
        show_header: Whether to show the "Export Data" header (default: True)
    """
    if df is None or df.empty:
        st.warning("No data available to download.")
        return
    
    if show_header:
        st.divider()
        st.subheader("Export/Preview Data")
    
    if show_data:
        st.dataframe(df.head(5000))

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download as CSV",
        csv,
        f"{filename_prefix}.csv",
        "text/csv",
        use_container_width=True
    )
    
