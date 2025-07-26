"""
Prompt templates for generating LLM responses
"""

# 🔹 QA-style prompt (no context)
QA_PROMPT_TEMPLATE = """You are a helpful and concise assistant.
Answer the following question as accurately as possible.

Question: {question}
Answer:"""

# 🔹 RAG-style prompt (question + retrieved context)
RAG_PROMPT_TEMPLATE = """You are an AI assistant that answers questions based on the provided context.
Use only the given context to answer. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""