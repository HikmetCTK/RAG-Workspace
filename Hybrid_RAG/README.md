# Hybrid RAG - Powerful Approach

🤝 What is Hybrid RAG?
A Hybrid RAG system combines two or more retrieval strategies to get the best of both worlds:

1. 🔍 BM25 Retriever (keyword-based):
Matches exact or partial terms from the query.

Fast and great for precision when the query has specific terms.

Doesn’t understand semantics — e.g., it treats “car” and “automobile” as unrelated.

2. 🤖 Vector Retriever (semantic-based using embeddings):
Uses vector embeddings to understand meaning and context.

Even if a query doesn't match exact words, it can find semantically similar content.

But sometimes may return slightly off results if embeddings aren’t perfect.

🔧 Technical Flow

User Query
   ↓
Hybrid Retriever (BM25 + Semantic)
   ↓
Top Document Chunks
   ↓
LLM (Gemini, GPT, etc.)
   ↓
Generated Answer
