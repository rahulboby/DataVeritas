"""
DataVeritas AI Assistant — Streamlit UI Section

Renders the AI Assistant page with:
- Auto-generated dataset health summary
- Chat interface for natural language questions
- Suggested example questions
"""

import streamlit as st
from core.ai.ai_assistant import (
    extract_dataset_insights,
    insights_to_context,
    build_llm_context,
    ask_llm_stream,
    generate_ai_summary,
)


# ==================== EXAMPLE QUESTIONS ====================

EXAMPLE_QUESTIONS = [
    "Why is the trust score low?",
    "Which column has the biggest data quality problem?",
    "What should I fix first to improve dataset reliability?",
    "Are there privacy risks in this dataset?",
    "Summarize the consistency rule violations.",
    "How many duplicate records exist and why does it matter?",
]


# ==================== MAIN DISPLAY FUNCTION ====================

def displayAIAssistant(df):
    """Main entry point — called from dashboard.py page routing."""

    st.markdown("""
        <style>
        .ai-header {
            color: #13293d;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .ai-subtext {
            color: #516477;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.caption("Ask natural language questions about your dataset and receive AI-powered explanations based on the computed insights.")

    # ---- 1. Extract insights & build context (cached per dataset) ----
    if "ai_insights_context" not in st.session_state or st.session_state.get("ai_df_id") != id(df):
        with st.spinner("Extracting dataset insights..."):
            insights = extract_dataset_insights(df)
            context = build_llm_context(insights)
            st.session_state.ai_insights_context = context
            st.session_state.ai_insights = insights
            st.session_state.ai_df_id = id(df)
            # Persist insights globally in session state (Step 6)
            st.session_state["dataset_insights"] = insights

    context = st.session_state.ai_insights_context
    insights = st.session_state.ai_insights

    # ---- 2. Quick metrics bar ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trust Score", f"{insights['trust_score']}%")
    m2.metric("Completeness", f"{insights['completeness_score']}%")
    m3.metric("Duplicates", f"{insights['duplicate_records']:,}")
    m4.metric("Outlier Rows", f"{insights['outlier_row_count']:,}")

    st.divider()

    # ---- 3. AI-generated summary ----
    st.markdown('<div class="ai-header">📊 AI Health Summary</div>', unsafe_allow_html=True)

    if "ai_summary" not in st.session_state:
        st.session_state.ai_summary = None

    if st.session_state.ai_summary:
        st.markdown(st.session_state.ai_summary)
    else:
        if st.button("Generate AI Summary", type="primary"):
            try:
                with st.spinner("Generating AI summary..."):
                    summary = generate_ai_summary(context)
                    st.session_state.ai_summary = summary
                    st.rerun()
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Failed to generate summary: {str(e)}")

    st.divider()

    # ---- 4. Chat interface ----
    st.markdown('<div class="ai-header">💬 Ask the AI Assistant</div>', unsafe_allow_html=True)

    # Initialize chat history
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    # Render example question buttons
    st.markdown('<div class="ai-subtext">Try an example question:</div>', unsafe_allow_html=True)
    btn_cols = st.columns(3)
    selected_example = None
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with btn_cols[i % 3]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                selected_example = q

    st.markdown("")  # spacing

    # Render chat history
    for msg in st.session_state.ai_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (or example button click)
    user_input = st.chat_input("Ask about your dataset quality...")

    # If an example button was clicked, use it as input
    if selected_example:
        user_input = selected_example

    if user_input:
        # Add user message to history and display it
        st.session_state.ai_chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream the assistant response
        with st.chat_message("assistant"):
            try:
                response = st.write_stream(ask_llm_stream(user_input, context))
                st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            except ValueError as e:
                error_msg = str(e)
                st.error(error_msg)
                st.session_state.ai_chat_history.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
            except Exception as e:
                error_msg = f"LLM request failed: {str(e)}"
                st.error(error_msg)
                st.session_state.ai_chat_history.append({"role": "assistant", "content": f"⚠️ {error_msg}"})

    # ---- 5. Sidebar-style controls ----
    with st.expander("Raw Insights Context (Debug)", expanded=False):
        st.code(context, language="text")

    if st.button("Clear Chat History"):
        st.session_state.ai_chat_history = []
        st.session_state.ai_summary = None
        st.rerun()
