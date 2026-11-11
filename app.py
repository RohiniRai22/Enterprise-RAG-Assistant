
import streamlit as st
from src.rag_pipeline import RAGPipeline


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# LIGHT PROFESSIONAL THEME
# ==================================================

st.markdown("""
<style>

/* Main page */
.stApp {
    background-color: #f4f7fc;
    color: #1e293b;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Sidebar text */
[data-testid="stSidebar"] p {
    color: #475569;
}

/* Main heading */
.main-heading {
    color: #172554;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 5px;
}

.main-subtitle {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Status cards */
.status-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

.status-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
}

.status-value {
    color: #172554;
    font-size: 17px;
    font-weight: 700;
    margin-top: 8px;
}

/* Welcome card */
.welcome-card {
    background: linear-gradient(120deg, #eaf2ff, #f2edff);
    border: 1px solid #dbeafe;
    border-radius: 18px;
    padding: 28px;
    margin-top: 20px;
    margin-bottom: 25px;
}

.welcome-card h2 {
    color: #172554;
    font-size: 25px;
}

.welcome-card p {
    color: #475569;
    font-size: 15px;
    line-height: 1.7;
}

/* Chat message containers */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
}

/* Chat text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #1e293b !important;
    font-size: 15px;
    line-height: 1.7;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
}

[data-testid="stChatInput"] textarea {
    color: #1e293b !important;
}

/* Buttons */
.stButton > button {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 15px;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: #ffffff;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #1d4ed8;
    font-weight: 600;
}

/* Divider */
hr {
    border-color: #e2e8f0;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD RAG PIPELINE
# ==================================================

@st.cache_resource
def load_pipeline():
    return RAGPipeline()


try:
    pipeline = load_pipeline()

except Exception as error:
    st.error(f"Could not initialize RAG pipeline: {error}")
    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🤖 Enterprise AI")

    st.caption("Your intelligent document assistant")

    st.divider()

    st.markdown("### Workspace")

    st.markdown("📚 **Knowledge Base**")
    st.caption("Search across your internal documents.")

    st.markdown("💬 **AI Assistant**")
    st.caption("Ask questions and receive document-based answers.")

    st.markdown("🔎 **Semantic Search**")
    st.caption("Find relevant information using AI embeddings.")

    st.divider()

    st.markdown("### Conversation")

    st.caption(f"Messages: {len(st.session_state.messages)}")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### System Status")

    st.success("● RAG Pipeline Ready")

    st.divider()

    st.caption("Enterprise RAG Assistant")
    st.caption("Powered by Retrieval-Augmented Generation")


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    """
    <div class="main-heading">
        🤖 Enterprise RAG Assistant
    </div>
    <div class="main-subtitle">
        Intelligent answers from your organization's internal documents.
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# STATUS CARDS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">SYSTEM STATUS</div>
            <div class="status-value">🟢 Online</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">SEARCH ENGINE</div>
            <div class="status-value">FAISS Vector Search</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">AI ENGINE</div>
            <div class="status-value">Gemini</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# WELCOME SECTION
# ==================================================

if not st.session_state.messages:

    st.markdown("")

    with st.container(border=True):

        st.markdown("## 👋 Welcome to your AI workspace")

        st.write(
            "Ask questions about company policies, employee benefits, "
            "travel guidelines, and IT support."
        )

        st.write(
            "Your answers are generated using information retrieved "
            "from your organization's internal documents."
        )

    st.markdown("### ✨ Start a conversation")

    st.caption("Try asking something like:")

    examples_col1, examples_col2 = st.columns(2)

    with examples_col1:
        st.markdown("- What is the annual leave entitlement?")
        st.markdown("- How many sick leave days are available?")

    with examples_col2:
        st.markdown("- What is the casual leave policy?")
        st.markdown("- Who is eligible for employee benefits?")


# ==================================================
# DISPLAY PREVIOUS MESSAGES
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 View Retrieved Sources"):

                for source in message["sources"]:
                    st.write(source)


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask anything about your company documents..."
)


# ==================================================
# PROCESS QUESTION
# ==================================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🔍 Searching documents and generating answer..."):

            try:

                result = pipeline.ask(question)

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("📚 View Retrieved Sources"):

                        for source in sources:
                            st.write(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                st.error(f"Something went wrong: {error}")
