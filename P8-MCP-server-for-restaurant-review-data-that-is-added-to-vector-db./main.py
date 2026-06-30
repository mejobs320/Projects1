"""
Restaurant Review RAG — Streamlit Web UI
Pipeline  : ChromaDB (RAG for restaurant reviews) → Tavily Web Search (anything)
Memory    : SQLite persistent storage
LLM       : Llama3.2 via Ollama (local)
"""

import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from vector import retriever
from memory import save_conversation, get_history, clear_history

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — paste your Tavily key here
# ═══════════════════════════════════════════════════════════════════════════

TAVILY_API_KEY = "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"   # ← paste your key here


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL & PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    return OllamaLLM(model="llama3.2")

model = load_model()

# Prompt when restaurant reviews are found
RESTAURANT_TEMPLATE = """
You are a helpful assistant that answers questions about a restaurant using customer reviews and web search results.

Relevant customer reviews from our database:
{reviews}

Additional web search context:
{web_context}

Answer the question below. Prioritise the customer reviews if they contain the answer.
If reviews don't cover it, use the web context.
Be friendly, helpful and concise.

Question: {question}
"""

# Prompt for general questions (no restaurant context needed)
GENERAL_TEMPLATE = """
You are a helpful assistant. Answer the following question using the web search results provided.
Be accurate, concise and helpful.

Web search results:
{web_context}

Question: {question}
"""

restaurant_prompt = ChatPromptTemplate.from_template(RESTAURANT_TEMPLATE)
general_prompt    = ChatPromptTemplate.from_template(GENERAL_TEMPLATE)

restaurant_chain  = restaurant_prompt | model
general_chain     = general_prompt    | model


# ═══════════════════════════════════════════════════════════════════════════
#  TAVILY WEB SEARCH
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_tavily_client():
    return TavilyClient(api_key=TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily and return combined context."""
    try:
        client  = get_tavily_client()
        results = client.search(query, max_results=max_results)
        snippets = []
        for r in results.get("results", []):
            snippets.append(f"- {r.get('title', '')}: {r.get('content', '')}")
        return "\n".join(snippets) if snippets else "No web results found."
    except Exception as e:
        return f"Web search unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════════════
#  QUERY PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

RESTAURANT_KEYWORDS = [
    "pizza", "pasta", "food", "menu", "dish", "restaurant", "service",
    "waiter", "ambience", "price", "reservation", "taste", "chef",
    "review", "rating", "portion", "delivery", "order", "dining",
    "dessert", "appetizer", "drink", "beer", "wine", "staff", "table",
]

def is_restaurant_query(question: str) -> bool:
    """Check if the question is likely about the restaurant."""
    q = question.lower()
    return any(kw in q for kw in RESTAURANT_KEYWORDS)


def run_query(question: str) -> tuple[str, str, str, str]:
    """
    Full pipeline:
    - Restaurant question → RAG + Web Search → Llama3.2
    - General question   → Web Search only   → Llama3.2
    Returns (answer, rag_context, web_context, source)
    """
    web_context  = web_search(question)
    rag_context  = ""
    source       = "google"

    if is_restaurant_query(question):
        # Pull from ChromaDB
        rag_docs    = retriever.invoke(question)
        rag_context = "\n\n".join([d.page_content for d in rag_docs]) if rag_docs else ""

        if rag_context:
            source = "both"
            answer = restaurant_chain.invoke({
                "reviews":     rag_context,
                "web_context": web_context,
                "question":    question,
            })
        else:
            source = "google"
            answer = general_chain.invoke({
                "web_context": web_context,
                "question":    question,
            })
    else:
        # General question — web search only
        answer = general_chain.invoke({
            "web_context": web_context,
            "question":    question,
        })

    return answer, rag_context, web_context, source


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="AI Assistant — Restaurant & Web",
        page_icon="🍕",
        layout="wide",
    )

    st.title("🍕 AI Assistant")
    st.caption("Restaurant reviews (RAG) + Web search (Tavily) + Llama3.2 (local)")

    # ── Sidebar — History ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("💬 Conversation History")

        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.success("History cleared!")
            st.rerun()

        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("📚 Restaurant questions → ChromaDB reviews")
        st.markdown("🌐 All questions → Tavily web search")
        st.markdown("🤖 Llama3.2 combines and answers")
        st.markdown("---")

        history = get_history(limit=30)
        if not history:
            st.caption("No conversations yet.")
        else:
            for item in reversed(history):
                with st.expander(f"Q: {item['question'][:40]}…", expanded=False):
                    st.markdown(f"**Source:** `{item['source']}`")
                    st.markdown(f"**Time:** {item['timestamp'][:19]}")
                    st.markdown(f"**Answer:** {item['answer'][:300]}…")

    # ── Main — Query input ────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "Ask anything:",
            placeholder="e.g. How is the pizza? OR What is the capital of France?",
        )
    with col2:
        submit = st.button("🔍 Ask", use_container_width=True)

    # ── Suggested questions ───────────────────────────────────────────────
    st.markdown("**Quick questions:**")
    q_cols = st.columns(4)
    suggestions = [
        "How is the pizza?",
        "Is the service good?",
        "Latest AI news?",
        "Best dishes to order?",
    ]
    for i, s in enumerate(suggestions):
        if q_cols[i].button(s, use_container_width=True):
            question = s
            submit   = True

    # ── Run pipeline ──────────────────────────────────────────────────────
    if submit and question:
        with st.spinner("Searching and thinking…"):
            answer, rag_context, web_context, source = run_query(question)
            save_conversation(question, answer, source)

        st.markdown("---")
        st.subheader("💡 Answer")
        st.markdown(answer)

        source_labels = {
            "rag":    "📚 Answered from restaurant reviews",
            "google": "🌐 Answered from web search",
            "both":   "📚🌐 Answered from reviews + web search",
        }
        st.caption(source_labels.get(source, source))

        if rag_context:
            with st.expander("📚 Retrieved Reviews (RAG context)"):
                st.text(rag_context)

        with st.expander("🌐 Tavily Web Search Results"):
            st.text(web_context)

    # ── Recent conversations ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🕐 Recent Conversations")
    history = get_history(limit=5)
    if not history:
        st.caption("No conversations yet. Ask your first question above!")
    else:
        for item in reversed(history):
            with st.expander(f"Q: {item['question']}", expanded=False):
                st.markdown(f"**A:** {item['answer']}")
                st.caption(f"Source: `{item['source']}` | {item['timestamp'][:19]}")


if __name__ == "__main__":
    main()
