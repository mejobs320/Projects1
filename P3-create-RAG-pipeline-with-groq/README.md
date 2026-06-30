**# project connect to  groq llm mode (llama-3.3-70b-versatile) and create RAG pipeline
**

create RAG with Groq models. 
data.txt file is given to LLM which is used to provide answer. whatever data is there in data.txt file, LLM will read and provide answer based 
on that data.


Model used and conncted:
create rag with Groq (llama-3.3-70b-versatile) — free tier
EMBED_MODEL   = "models/gemini-embedding-001"   # Google embedding
CHAT_MODEL    = "llama-3.3-70b-versatile"       # Groq LLM (free)

Code:
CHUNKING, EMBEDDINGS, VECTOR STORE , RAG PIPELINE is created

Run program python main2.py
:ingest data.txt

Result:
LLm will answer as below as same data is shared in data.txt
how many states in china
2 states


Note:
Google is now used only for embeddings (it's good at that, free tier is generous)
Groq handles all LLM responses — completely free, very fast (runs on custom hardware)
     Groq Model is llama-3.3-70b-versatile — a powerful open-source model


