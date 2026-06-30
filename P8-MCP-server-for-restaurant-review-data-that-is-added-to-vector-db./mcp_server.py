"""
MCP Server — Restaurant Review Assistant
Exposes tools for:
  1. RAG search via ChromaDB (restaurant reviews)
  2. SQLite review data (read-only: title, date, rating, review)
  3. Tavily web search (general + restaurant)

Transport: stdio (default) or SSE (--sse flag)
"""

import sys
import os
import json
import sqlite3
import argparse
import asyncio
from pathlib import Path

# ── MCP ──────────────────────────────────────────────────────────────────
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ── LangChain / ChromaDB ─────────────────────────────────────────────────
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ── Tavily ───────────────────────────────────────────────────────────────
from tavily import TavilyClient

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

TAVILY_API_KEY = "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"
EMBED_MODEL    = "mxbai-embed-large"

import sys
from pathlib import Path
BASE_DIR   = Path(__file__).parent.resolve()
DB_PATH    = str(BASE_DIR / "conversation_memory.db")
CSV_PATH   = str(BASE_DIR / "realistic_restaurant_reviews.csv")
CHROMA_DIR = str(BASE_DIR / "chroma_langchain_db")


# ═══════════════════════════════════════════════════════════════════════════
#  INIT CHROMADB RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════

def get_retriever():
    embeddings   = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── Tool 1: RAG semantic search ───────────────────────────────────────────
def tool_rag_search(query: str) -> str:
    """Semantic search over restaurant reviews using ChromaDB."""
    try:
        retriever = get_retriever()
        docs      = retriever.invoke(query)
        if not docs:
            return "No relevant reviews found for your query."
        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            results.append(
                f"[{i}] Rating: {meta.get('rating', 'N/A')} | Date: {meta.get('date', 'N/A')}\n"
                f"    {doc.page_content}"
            )
        return "\n\n".join(results)
    except Exception as e:
        return f"RAG search error: {e}"


# ── Tool 2: Get all reviews (paginated) ───────────────────────────────────
def tool_get_reviews(limit: int = 20, offset: int = 0, min_rating: float = 0) -> str:
    """Read reviews from SQLite. Filter by min_rating if provided."""
    try:
        con  = sqlite3.connect(DB_PATH)
        rows = con.execute(
            """SELECT title, date, rating, review FROM restaurant_reviews
               WHERE rating >= ? ORDER BY date DESC LIMIT ? OFFSET ?""",
            (min_rating, limit, offset),
        ).fetchall()
        con.close()
        if not rows:
            return "No reviews found."
        out = []
        for r in rows:
            out.append(f"Title : {r[0]}\nDate  : {r[1]}\nRating: {r[2]}\nReview: {r[3]}")
        return "\n\n---\n\n".join(out)
    except Exception as e:
        return f"DB error: {e}"


# ── Tool 3: Get review statistics ────────────────────────────────────────
def tool_get_stats() -> str:
    """Return summary statistics of all restaurant reviews."""
    try:
        con = sqlite3.connect(DB_PATH)
        stats = con.execute("""
            SELECT
                COUNT(*)            AS total,
                ROUND(AVG(rating),2) AS avg_rating,
                MAX(rating)          AS max_rating,
                MIN(rating)          AS min_rating,
                SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS negative
            FROM restaurant_reviews
        """).fetchone()
        con.close()
        return (
            f"Total reviews : {stats[0]}\n"
            f"Average rating: {stats[1]}\n"
            f"Highest rating: {stats[2]}\n"
            f"Lowest rating : {stats[3]}\n"
            f"Positive (≥4) : {stats[4]}\n"
            f"Negative (≤2) : {stats[5]}"
        )
    except Exception as e:
        return f"Stats error: {e}"


# ── Tool 4: Search reviews by rating ─────────────────────────────────────
def tool_reviews_by_rating(rating: int) -> str:
    """Get all reviews with an exact star rating (1-5)."""
    try:
        con  = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT title, date, rating, review FROM restaurant_reviews WHERE rating = ? ORDER BY date DESC",
            (rating,),
        ).fetchall()
        con.close()
        if not rows:
            return f"No reviews with rating {rating}."
        out = [f"Found {len(rows)} review(s) with rating {rating}:\n"]
        for r in rows:
            out.append(f"Title : {r[0]}\nDate  : {r[1]}\nReview: {r[3]}")
        return "\n\n---\n\n".join(out)
    except Exception as e:
        return f"DB error: {e}"


# ── Tool 5: Search reviews by keyword ────────────────────────────────────
def tool_search_reviews_by_keyword(keyword: str) -> str:
    """Search review titles and text for a keyword."""
    try:
        con  = sqlite3.connect(DB_PATH)
        rows = con.execute(
            """SELECT title, date, rating, review FROM restaurant_reviews
               WHERE title LIKE ? OR review LIKE ? ORDER BY date DESC LIMIT 20""",
            (f"%{keyword}%", f"%{keyword}%"),
        ).fetchall()
        con.close()
        if not rows:
            return f"No reviews found containing '{keyword}'."
        out = [f"Found {len(rows)} review(s) matching '{keyword}':\n"]
        for r in rows:
            out.append(f"Title : {r[0]}\nDate  : {r[1]}\nRating: {r[2]}\nReview: {r[3]}")
        return "\n\n---\n\n".join(out)
    except Exception as e:
        return f"DB error: {e}"


# ── Tool 6: Get conversation history ─────────────────────────────────────
def tool_get_conversation_history(limit: int = 10) -> str:
    """Retrieve recent conversation history from SQLite memory."""
    try:
        con  = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT timestamp, question, answer, source FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        con.close()
        if not rows:
            return "No conversation history found."
        out = []
        for r in rows:
            out.append(f"Time  : {r[0][:19]}\nSource: {r[3]}\nQ: {r[1]}\nA: {r[2]}")
        return "\n\n---\n\n".join(reversed(out))
    except Exception as e:
        return f"History error: {e}"


# ── Tool 7: Tavily web search ─────────────────────────────────────────────
def tool_web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily for any query."""
    try:
        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query, max_results=max_results)
        snippets = []
        for r in results.get("results", []):
            snippets.append(f"- {r.get('title','')}: {r.get('content','')}")
        return "\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Web search error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
#  INIT SQLITE — load CSV into DB if needed
# ═══════════════════════════════════════════════════════════════════════════

def init_sqlite():
    """Load CSV data into SQLite restaurant_reviews table if not already done."""
    con = sqlite3.connect(DB_PATH)

    # Create conversations table
    con.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, question TEXT, answer TEXT, source TEXT
        )
    """)

    # Create reviews table
    con.execute("""
        CREATE TABLE IF NOT EXISTS restaurant_reviews (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title  TEXT,
            date   TEXT,
            rating REAL,
            review TEXT
        )
    """)

    # Load CSV if table is empty
    count = con.execute("SELECT COUNT(*) FROM restaurant_reviews").fetchone()[0]
    if count == 0 and Path(CSV_PATH).exists():
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            con.execute(
                "INSERT INTO restaurant_reviews (title, date, rating, review) VALUES (?,?,?,?)",
                (row["Title"], row["Date"], row["Rating"], row["Review"]),
            )
        print(f"  [✓] Loaded {len(df)} reviews into SQLite", file=sys.stderr)

    con.commit()
    con.close()


# ═══════════════════════════════════════════════════════════════════════════
#  MCP SERVER
# ═══════════════════════════════════════════════════════════════════════════

server = Server("restaurant-review-mcp")

TOOLS = [
    Tool(
        name="rag_search",
        description="Semantic search over restaurant reviews using ChromaDB RAG pipeline. Use for natural language questions about food, service, ambience etc.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query about restaurant reviews"}
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_reviews",
        description="Get restaurant reviews from the database with optional rating filter and pagination.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit":      {"type": "integer", "description": "Number of reviews to return (default 20)"},
                "offset":     {"type": "integer", "description": "Pagination offset (default 0)"},
                "min_rating": {"type": "number",  "description": "Minimum star rating filter (1-5, default 0)"},
            },
        },
    ),
    Tool(
        name="get_review_stats",
        description="Get summary statistics of all restaurant reviews: total count, average rating, positive/negative breakdown.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="reviews_by_rating",
        description="Get all reviews filtered by an exact star rating (1-5).",
        inputSchema={
            "type": "object",
            "properties": {
                "rating": {"type": "integer", "description": "Star rating to filter by (1-5)"}
            },
            "required": ["rating"],
        },
    ),
    Tool(
        name="search_reviews_by_keyword",
        description="Search review titles and text for a specific keyword like 'pizza', 'service', 'price' etc.",
        inputSchema={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Keyword to search for in reviews"}
            },
            "required": ["keyword"],
        },
    ),
    Tool(
        name="get_conversation_history",
        description="Retrieve recent conversation history stored in SQLite memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Number of recent conversations to return (default 10)"}
            },
        },
    ),
    Tool(
        name="web_search",
        description="Search the web using Tavily for any general or restaurant-related query.",
        inputSchema={
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Search query"},
                "max_results": {"type": "integer", "description": "Number of results (default 5)"},
            },
            "required": ["query"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "rag_search":
        result = tool_rag_search(arguments.get("query", ""))

    elif name == "get_reviews":
        result = tool_get_reviews(
            limit=arguments.get("limit", 20),
            offset=arguments.get("offset", 0),
            min_rating=arguments.get("min_rating", 0),
        )

    elif name == "get_review_stats":
        result = tool_get_stats()

    elif name == "reviews_by_rating":
        result = tool_reviews_by_rating(arguments.get("rating", 5))

    elif name == "search_reviews_by_keyword":
        result = tool_search_reviews_by_keyword(arguments.get("keyword", ""))

    elif name == "get_conversation_history":
        result = tool_get_conversation_history(arguments.get("limit", 10))

    elif name == "web_search":
        result = tool_web_search(
            arguments.get("query", ""),
            arguments.get("max_results", 5),
        )
    else:
        result = f"Unknown tool: {name}"

    return [TextContent(type="text", text=result)]


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

async def run_stdio():
    print("  [MCP] Starting stdio server…", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def run_sse(host: str = "0.0.0.0", port: int = 8000):
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    import uvicorn

    sse       = SseServerTransport("/messages/")
    init_opts = server.create_initialization_options()

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
            await server.run(streams[0], streams[1], init_opts)

    async def handle_messages(scope, receive, send):
        await sse.handle_post_message(scope, receive, send)

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=handle_messages),
        ]
    )

    print(f"  [MCP] SSE server running at http://{host}:{port}/sse", file=sys.stderr)
    config = uvicorn.Config(starlette_app, host=host, port=port)
    await uvicorn.Server(config).serve()


def main():
    init_sqlite()

    parser = argparse.ArgumentParser(description="Restaurant Review MCP Server")
    parser.add_argument("--sse",  action="store_true", help="Run as SSE server instead of stdio")
    parser.add_argument("--host", default="0.0.0.0",   help="SSE host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="SSE port (default: 8000)")
    args = parser.parse_args()

    if args.sse:
        asyncio.run(run_sse(args.host, args.port))
    else:
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
