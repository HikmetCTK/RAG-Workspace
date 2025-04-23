# Hybrid RAG - Powerful Approach
![hybrid_rag](https://github.com/user-attachments/assets/2aeb1a3d-21bc-411c-b89d-ecb1eed862da)

# 🤝 What is Hybrid RAG?
A Hybrid RAG system combines two or more retrieval strategies to get the best of both worlds:

# 🔍 **BM25 Retriever (keyword-based):**

Matches exact or partial terms from the query.

Fast and great for precision when the query has specific terms.

Doesn’t understand semantics — e.g., it treats “car” and “automobile” as unrelated.

# 🤖 Vector Retriever (semantic-based using embeddings):
Uses vector embeddings to understand meaning and context.

# 🔧 Technical Flow

User Query ➡️ Hybrid Retriever (BM25 + Semantic) ➡️ Top Document Chunks ➡️ LLM (Gemini, GPT, etc.) ➡️ Generated Answer
