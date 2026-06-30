"""
Quick local test for MCP tools — runs without Claude Desktop.
Tests all 7 tools directly.
"""
import sys
sys.path.insert(0, ".")

# Import tool functions directly
from mcp_server import (
    init_sqlite,
    tool_rag_search,
    tool_get_reviews,
    tool_get_stats,
    tool_reviews_by_rating,
    tool_search_reviews_by_keyword,
    tool_get_conversation_history,
    tool_web_search,
)

def test(name, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    result = fn(*args, **kwargs)
    print(result[:500])   # show first 500 chars
    print("  [OK]")

if __name__ == "__main__":
    print("Initialising SQLite…")
    init_sqlite()

    test("RAG Search — pizza",          tool_rag_search, "how is the pizza?")
    test("Get Reviews (top 3)",         tool_get_reviews, 3, 0, 0)
    test("Review Stats",                tool_get_stats)
    test("Reviews by Rating (5 stars)", tool_reviews_by_rating, 5)
    test("Keyword Search — service",    tool_search_reviews_by_keyword, "service")
    test("Conversation History",        tool_get_conversation_history, 5)
    test("Web Search — pizza recipes",  tool_web_search, "best pizza restaurants 2025", 3)

    print(f"\n{'='*60}")
    print("All tests passed!")
