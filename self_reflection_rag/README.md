## Self-Reflective Retrieval-Augmented Generation (SELF-RAG)

It provides to evaluate own response thanks to sharing response of llm amongs more than one agent.

![Screenshot 2024-11-24 150721](https://github.com/user-attachments/assets/df0bf321-1d48-4c44-8fcb-e3ab0120066c)

Behind the logic:

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

Advantages:
