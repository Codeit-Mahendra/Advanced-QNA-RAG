
from langchain_core.prompts import ChatPromptTemplate
#chat model has two input system & User prompt

system_prompt = """Answer the question based ONLY on the provided context.

CRITICAL RULES:
1. Provide ONLY the final answer - no internal reasoning
2. NEVER use <think> tags or any XML tags in your response
3. NEVER say "based on the context" or similar phrases
4. If you're unsure, say "I cannot find this information"

Context: {context}
Question: {input}
"""



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)