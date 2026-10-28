"""
DataVeritas AI Assistant — Streamlit UI Section

Renders the AI Assistant page with:
- Auto-generated dataset health summary
- Chat interface for natural language questions
- Suggested example questions
"""

import html

import streamlit as st
from core.ai.DQ_Copilot import (
    extract_dataset_insights,
    insights_to_context,
    build_llm_context,
    ask_llm,
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
    "Which 3 columns are best predictors for regression modeling?",
    "Which columns have skewed distributions and need transformation?",
    "Identify columns with high correlation for feature selection.",
    "What are the most reliable features for machine learning?",
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
            if st.button(q, key=f"example_{i}", width='stretch'):
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


@st.fragment
def displayFloatingAIAssistant(df):
    """Render a bottom-right Copilot widget that can be opened from any dashboard page."""

    apply_floating_copilot_css()

    st.session_state.setdefault("floating_copilot_open", False)
    st.session_state.setdefault("floating_copilot_messages", [])

    with st.container(key="floating_copilot_shell"):
        if not st.session_state.floating_copilot_open:
            with st.container(key="floating_copilot_launcher"):
                open_clicked = st.button("Ask Copilot", key="floating_copilot_open_btn", type="primary")

            if open_clicked:
                st.session_state.floating_copilot_open = True
                st.rerun(scope="fragment")
            else:
                return

        with st.container(key="floating_copilot_panel"):
            header_col, close_col = st.columns([5, 1], vertical_alignment="center")

            with header_col:
                st.markdown(
                    f"""
                    <div class="dv-copilot-header">
                        <div class="dv-copilot-title">DataVeritas Copilot</div>
                        <div class="dv-copilot-subtitle">
                            Using {len(df):,} rows across {len(df.columns):,} columns
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with close_col:
                if st.button("X", key="floating_copilot_close_btn", width="stretch"):
                    st.session_state.floating_copilot_open = False
                    st.rerun(scope="fragment")

            with st.container(key="floating_copilot_messages_box"):
                if not st.session_state.floating_copilot_messages:
                    render_empty_copilot_state()

                for idx, msg in enumerate(st.session_state.floating_copilot_messages[-8:]):
                    render_floating_copilot_message(msg["role"], msg["content"], idx)

            with st.form("floating_copilot_chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Message",
                    placeholder="Ask about trust score, missing values, duplicates, outliers...",
                    height=90,
                    label_visibility="collapsed",
                )
                send = st.form_submit_button("Send", type="primary", width="stretch")

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                summarize = st.button("Summarize Issues", key="floating_copilot_summarize", width="stretch")
            with action_col2:
                clear = st.button("Clear", key="floating_copilot_clear", width="stretch")

            if clear:
                st.session_state.floating_copilot_messages = []
                st.rerun(scope="fragment")

            if summarize:
                user_input = "Summarize the most important data quality issues and what to fix first."

            if send and user_input:
                handle_floating_copilot_message(user_input, df)
                st.rerun(scope="fragment")

            if summarize:
                handle_floating_copilot_message(user_input, df)
                st.rerun(scope="fragment")


def handle_floating_copilot_message(user_input: str, df):
    st.session_state.floating_copilot_messages.append({"role": "user", "content": user_input})

    try:
        context, _insights = get_floating_copilot_context(df)
        with st.spinner("Copilot is thinking..."):
            response = ask_llm(user_input, context)
    except ValueError as exc:
        response = str(exc)
    except Exception as exc:
        response = f"Copilot request failed: {exc}"

    st.session_state.floating_copilot_messages.append({"role": "assistant", "content": response})


def render_empty_copilot_state():
    st.markdown(
        """
        <div class="dv-copilot-empty">
            <div class="dv-copilot-empty-title">Ask about this dataset</div>
            <div class="dv-copilot-empty-copy">
                Try asking what is hurting the trust score, which fields need cleanup, or what to fix first.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_floating_copilot_message(role: str, content: str, message_idx: int):
    role_class = "user" if role == "user" else "assistant"
    label = "You" if role_class == "user" else "Copilot"

    if role_class == "user":
        safe_content = html.escape(content).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="dv-chat-row user">
                <div class="dv-chat-bubble">
                    <div class="dv-chat-label">{label}</div>
                    <div class="dv-chat-content">{safe_content}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.container(key=f"floating_copilot_msg_assistant_{message_idx}"):
        st.markdown('<div class="dv-chat-label">Copilot</div>', unsafe_allow_html=True)
        st.markdown(content)


def get_floating_copilot_context(df):
    signature = build_df_signature(df)

    if (
        st.session_state.get("floating_copilot_df_signature") != signature
        or "floating_copilot_context" not in st.session_state
        or "floating_copilot_insights" not in st.session_state
    ):
        with st.spinner("Preparing Copilot context..."):
            insights = extract_dataset_insights(df)
            context = build_llm_context(insights)
            st.session_state.floating_copilot_df_signature = signature
            st.session_state.floating_copilot_context = context
            st.session_state.floating_copilot_insights = insights

    return st.session_state.floating_copilot_context, st.session_state.floating_copilot_insights


def build_df_signature(df):
    return (
        df.shape,
        tuple(df.columns),
        tuple(str(dtype) for dtype in df.dtypes),
        int(df.isna().sum().sum()),
    )


def apply_floating_copilot_css():
    st.markdown(
        """
        <style>
        .st-key-floating_copilot_shell {
            position: fixed;
            right: 1.25rem;
            bottom: 1.25rem;
            z-index: 99999;
            width: auto;
            max-width: calc(100vw - 2rem);
        }

        .st-key-floating_copilot_shell > div {
            display: flex;
            justify-content: flex-end;
        }

        .st-key-floating_copilot_launcher button {
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.35);
            background: #0f2f45;
            color: #ffffff;
            font-weight: 700;
            padding: 0.72rem 1rem;
            box-shadow: 0 12px 30px rgba(15, 47, 69, 0.28);
        }

        .st-key-floating_copilot_panel {
            background: #ffffff;
            border: 1px solid rgba(15, 47, 69, 0.14);
            border-radius: 8px;
            box-shadow: 0 20px 54px rgba(15, 47, 69, 0.24);
            padding: 0.85rem;
            width: min(390px, calc(100vw - 2rem));
            max-height: min(660px, calc(100vh - 2.5rem));
            overflow-y: auto;
        }

        .st-key-floating_copilot_panel h4,
        .st-key-floating_copilot_panel p {
            margin: 0;
        }

        .dv-copilot-header {
            padding: 0.15rem 0 0.35rem 0;
        }

        .dv-copilot-title {
            color: #13293d;
            font-size: 1rem;
            font-weight: 750;
            line-height: 1.2;
        }

        .dv-copilot-subtitle {
            color: #64748b;
            font-size: 0.78rem;
            line-height: 1.25;
            margin-top: 0.18rem;
        }

        .st-key-floating_copilot_panel button {
            border-radius: 8px;
        }

        .st-key-floating_copilot_close_btn button {
            min-height: 2rem;
            padding: 0;
            border-radius: 999px;
        }

        .st-key-floating_copilot_messages_box {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.65rem;
            background: #ffffff;
            height: 310px;
            overflow-y: auto;
            margin: 0.35rem 0 0.7rem 0;
        }

        .dv-copilot-empty {
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            padding: 0.8rem;
            background: #ffffff;
            color: #475569;
        }

        .dv-copilot-empty-title {
            color: #13293d;
            font-weight: 700;
            font-size: 0.92rem;
            margin-bottom: 0.25rem;
        }

        .dv-copilot-empty-copy {
            font-size: 0.82rem;
            line-height: 1.35;
        }

        .dv-chat-row {
            display: flex;
            margin: 0.45rem 0;
        }

        .dv-chat-row.user {
            justify-content: flex-end;
        }

        .dv-chat-row.assistant {
            justify-content: flex-start;
        }

        [class*="st-key-floating_copilot_msg_assistant"] {
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
            border-radius: 8px;
            padding: 0.58rem 0.68rem;
            margin: 0.45rem 0;
            background: #ffffff;
            color: #13293d;
            border: 1px solid #e2e8f0;
            font-size: 0.86rem;
            line-height: 1.35;
            overflow-x: auto;
        }

        [class*="st-key-floating_copilot_msg_assistant"] p,
        [class*="st-key-floating_copilot_msg_assistant"] ul,
        [class*="st-key-floating_copilot_msg_assistant"] ol {
            margin-bottom: 0.4rem;
        }

        [class*="st-key-floating_copilot_msg_assistant"] [data-testid="stMarkdownContainer"] {
            overflow-x: auto;
        }

        [class*="st-key-floating_copilot_msg_assistant"] table {
            display: block;
            max-width: 100%;
            overflow-x: auto;
            border-collapse: collapse;
            background: #ffffff;
            font-size: 0.78rem;
        }

        [class*="st-key-floating_copilot_msg_assistant"] th,
        [class*="st-key-floating_copilot_msg_assistant"] td {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 0.35rem 0.45rem;
            white-space: nowrap;
        }

        [class*="st-key-floating_copilot_msg_assistant"] th {
            background: #f8fafc;
            color: #13293d;
            font-weight: 700;
        }

        [class*="st-key-floating_copilot_msg_assistant"] h1,
        [class*="st-key-floating_copilot_msg_assistant"] h2,
        [class*="st-key-floating_copilot_msg_assistant"] h3 {
            font-size: 0.95rem;
            margin: 0.2rem 0 0.35rem 0;
        }

        .dv-chat-bubble {
            max-width: 84%;
            border-radius: 8px;
            padding: 0.58rem 0.68rem;
            font-size: 0.86rem;
            line-height: 1.35;
            word-wrap: break-word;
        }

        .dv-chat-row.user .dv-chat-bubble {
            background: #1f5a7a;
            color: #ffffff;
        }

        .dv-chat-label {
            font-size: 0.68rem;
            font-weight: 750;
            opacity: 0.75;
            margin-bottom: 0.18rem;
        }

        .dv-chat-content {
            white-space: normal;
        }

        .st-key-floating_copilot_chat_form textarea {
            min-height: 74px !important;
            font-size: 0.88rem;
        }

        .st-key-floating_copilot_chat_form button {
            background: #1f5a7a;
            border-color: #1f5a7a;
            color: #ffffff;
        }

        @media (max-width: 640px) {
            .st-key-floating_copilot_shell {
                right: 0.75rem;
                bottom: 0.75rem;
                max-width: calc(100vw - 1.5rem);
            }

            .st-key-floating_copilot_panel {
                width: calc(100vw - 1.5rem);
                max-height: calc(100vh - 1.5rem);
            }

            .st-key-floating_copilot_messages_box {
                height: 300px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
