## Hypothetical Document Embedding (HyDE) for RAG 💡🎯
Behind the logic: In  RAG ,user question is searched in document and get chunks based on similarity score . Then We give these chunks with user question to llm and
LLM gives answer.

HyDE:First user question is asked to LLM and LLM's answer searched in documents then gets chunks based on LLM answer .Finally, Chunks and user question are given to llm.
LLM gives answer based on chunks.

![Ekran görüntüsü 2024-11-23 172544](https://github.com/user-attachments/assets/34c8a59c-47e4-4775-9bc4-2fb1f2b76587)

Key Advantages 🔎: 
-This approach helps to get more context-aware answer

-Robust to  question that lacks specificity or lacks easily identifiable elements to derive an answer from a given context,

-Reduce hallucinations because HyDE first generates a general hypothetical response and retrieves authoritative articles to base the final answer on factual content.

-Zero-Shot Retrieval: HyDE can work “out of the box” without relying on a large dataset of labeled examples.

