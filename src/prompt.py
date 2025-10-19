
#chat model has two input system & User prompt

system_prompt = (
    "You are the best assistant for answering questions based on the provided context. "
    "Use the context below to answer accurately and factually. "
    "If you don't know the answer, say clearly You don't know. "
    "Limit your response to two sentences.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)