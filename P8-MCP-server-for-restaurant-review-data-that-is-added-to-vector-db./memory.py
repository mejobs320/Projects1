# memory.py — Persistent conversation memory using SQLite

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "conversation_memory.db"


def init_db():
    """Create tables if they don't exist."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question  TEXT,
            answer    TEXT,
            source    TEXT
        )
    """)
    con.commit()
    con.close()


def save_conversation(question: str, answer: str, source: str = "rag"):
    """Save a Q&A pair to memory. source = 'rag' | 'google' | 'both'"""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO conversations (timestamp, question, answer, source) VALUES (?,?,?,?)",
        (datetime.now().isoformat(), question, answer, source),
    )
    con.commit()
    con.close()


def get_history(limit: int = 50) -> list[dict]:
    """Return last N conversations."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, timestamp, question, answer, source FROM conversations ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    return [
        {"id": r[0], "timestamp": r[1], "question": r[2], "answer": r[3], "source": r[4]}
        for r in reversed(rows)
    ]


def clear_history():
    """Wipe all conversation history."""
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM conversations")
    con.commit()
    con.close()


# Initialise on import
init_db()
