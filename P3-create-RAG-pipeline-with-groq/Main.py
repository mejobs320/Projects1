"""
RAG Pipeline — Terminal Interface
Uses google-genai SDK (new) for embeddings + Gemini LLM
"""

import os
import sys
import pickle
import textwrap
from pathlib import Path

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    import numpy as np
    import faiss
except ImportError:
    print("Missing dependencies. Run:  pip install -r requirements.txt")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

CHUNK_SIZE   = 500
CHUNK_OVERLAP = 100
TOP_K        = 4
EMBED_MODEL  = "models/gemini-embedding-001"
CHAT_MODEL   = "gemini-2.0-flash"
INDEX_FILE   = "rag_index.pkl"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided context. If the answer is not found in the context, say so honestly "
    "— do not make things up. Be concise and precise."
)


# ═══════════════════════════════════════════════════════════════════════════
#  CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_documents(paths: list[str]) -> list[dict]:
    docs = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"  [!] File not found: {path}")
            continue
        if path.suffix.lower() != ".txt":
            print(f"  [!] Skipping non-.txt file: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        docs.append({"source": str(path), "text": text})
        print(f"  [+] Loaded: {path}  ({len(text):,} chars)")
    return docs


# ═══════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    vectors = []
    for i, text in enumerate(texts, 1):
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        vectors.append(response.embeddings[0].values)
        if i % 10 == 0 or i == len(texts):
            print(f"  Embedded {i}/{len(texts)} chunks…", end="\r")
    print()
    return np.array(vectors, dtype="float32")


def embed_single(client: genai.Client, text: str) -> np.ndarray:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return np.array([response.embeddings[0].values], dtype="float32")


# ═══════════════════════════════════════════════════════════════════════════
#  VECTOR STORE (FAISS)
# ═══════════════════════════════════════════════════════════════════════════

class VectorStore:
    def __init__(self):
        self.index   = None
        self.chunks  = []
        self.sources = []

    def build(self, chunks, sources, vectors):
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks  = chunks
        self.sources = sources
        print(f"  [✓] FAISS index built — {self.index.ntotal} vectors (dim={dim})")

    def search(self, query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "text":   self.chunks[idx],
                    "source": self.sources[idx],
                    "score":  float(score),
                })
        return results

    def save(self, path: str = INDEX_FILE):
        data = {
            "chunks":  self.chunks,
            "sources": self.sources,
            "vectors": faiss.serialize_index(self.index),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  [✓] Index saved → {path}")

    def load(self, path: str = INDEX_FILE) -> bool:
        if not Path(path).exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks  = data["chunks"]
        self.sources = data["sources"]
        self.index   = faiss.deserialize_index(data["vectors"])
        print(f"  [✓] Index loaded ← {path}  ({self.index.ntotal} vectors)")
        return True


# ═══════════════════════════════════════════════════════════════════════════
#  RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
        self.store  = VectorStore()

    def ingest(self, file_paths: list[str]):
        print("\n── Ingesting documents ──────────────────────────────")
        docs = load_documents(file_paths)
        if not docs:
            print("  [!] No valid documents loaded.")
            return

        all_chunks, all_sources = [], []
        for doc in docs:
            chunks = chunk_text(doc["text"])
            all_chunks.extend(chunks)
            all_sources.extend([doc["source"]] * len(chunks))
            print(f"  → {doc['source']}: {len(chunks)} chunks")

        print(f"\n  Total chunks: {len(all_chunks)}")
        print("  Generating embeddings…")
        vectors = embed_texts(self.client, all_chunks)

        self.store.build(all_chunks, all_sources, vectors)
        self.store.save()

    def query(self, question: str) -> str:
        query_vec = embed_single(self.client, question)
        hits = self.store.search(query_vec, top_k=TOP_K)
        if not hits:
            return "No relevant context found in the indexed documents."

        context = "\n\n---\n\n".join(
            f"[Source {i}: {Path(h['source']).name}  score={h['score']:.3f}]\n{h['text']}"
            for i, h in enumerate(hits, 1)
        )

        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        response = self.client.models.generate_content(
            model=CHAT_MODEL,
            contents=full_prompt,
        )
        return response.text.strip()

    def load_index(self) -> bool:
        print("\n── Loading existing index ───────────────────────────")
        return self.store.load()


# ═══════════════════════════════════════════════════════════════════════════
#  TERMINAL REPL
# ═══════════════════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════╗
║          RAG Pipeline  —  Terminal Interface         ║
║   Powered by Google GenAI (new SDK) + FAISS          ║
╚══════════════════════════════════════════════════════╝

Commands:
  :ingest <file1.txt> [file2.txt …]   — index new documents
  :sources                            — list indexed sources
  :clear                              — wipe index
  :help                               — show this message
  :quit  /  exit                      — exit

Type any question to query the indexed documents.
"""

def wrap(text: str, width: int = 80) -> str:
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())


def repl(pipeline: RAGPipeline):
    print(BANNER)
    if not pipeline.load_index():
        print("  No saved index found. Use  :ingest <files>  to add documents.\n")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in (":quit", "exit", "quit"):
            print("Bye!")
            break

        elif user_input.lower() == ":help":
            print(BANNER)

        elif user_input.lower() == ":sources":
            if not pipeline.store.chunks:
                print("  No documents indexed yet.")
            else:
                unique = sorted(set(pipeline.store.sources))
                print(f"\n  Indexed sources ({len(unique)}):")
                for s in unique:
                    print(f"    • {s}  ({pipeline.store.sources.count(s)} chunks)")

        elif user_input.lower() == ":clear":
            pipeline.store = VectorStore()
            if Path(INDEX_FILE).exists():
                Path(INDEX_FILE).unlink()
            print("  [✓] Index cleared.")

        elif user_input.startswith(":ingest"):
            parts = user_input.split()[1:]
            if not parts:
                print("  Usage:  :ingest <file1.txt> [file2.txt …]")
            else:
                pipeline.ingest(parts)

        else:
            if not pipeline.store.chunks:
                print("  [!] No documents indexed. Use  :ingest <files>  first.")
                continue
            print("\n  Thinking…")
            try:
                answer = pipeline.query(user_input)
                print("\n" + "─" * 60)
                print(wrap(answer))
                print("─" * 60)
            except Exception as e:
                print(f"  [Error] {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("\n[Error] GOOGLE_API_KEY not found. Create a .env file with:\n")
        print("  GOOGLE_API_KEY=your-key-here\n")
        sys.exit(1)

    pipeline = RAGPipeline(api_key=api_key)
    repl(pipeline)


if __name__ == "__main__":
    main()
