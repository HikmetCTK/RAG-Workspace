## Self-Reflective Retrieval-Augmented Generation (SELF-RAG) ![robot_1f916](https://github.com/user-attachments/assets/416028a9-8a94-4a8c-a1ec-d1f7b03a4c53)

Self-Retrieval-Augmented Generation (Self-RAG) is an advanced RAG paradigm where the language model (LLM) is directly responsible for both the retrieval of relevant information and the generation of responses. Unlike standard RAG.

It provides to evaluate own response thanks to sharing response of llm amongs more than one agent.

![Screenshot 2024-11-24 150721](https://github.com/user-attachments/assets/df0bf321-1d48-4c44-8fcb-e3ab0120066c)

Behind the logic:🧩🚀

First, like normal rag implenetation it answers user question. Then another agent evaluate response of llm  based on query, context  and response 
Evaluation metrics:
-relevance
-factual_accuracy
-completeness
-coherence

At the end, we take overall score and overall_explanation about response from this agent as json format.
If avg. score is above 8 (our threshold for this project), we will get response.
Otherwise, response,explanation, and query are sent to another agent which is tasked  to improve our response based on feedback.
Finally, we get our response which is evaluated and confirmed.

# Advantages:✅

-Enhanced Factual Accuracy: One of the primary advantages of Self-RAG is its ability to improve the factual accuracy of LLM outputs.

-Improved Relevance and Quality: Higher quality outputs that are more aligned with the user's needs and the task requirements.

-Versatility and Adaptability:This adaptability makes Self-RAG suitable for a wide range of applications, including open-domain question answering, reasoning, and fact verification tasks. As a result, it offers a robust solution for various real-world challenges.
