import os
from typing import Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from graph.state import GraphState

#  the default length of the embedding vector is 1536 for text-embedding-3-small
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    dimensions=512,
)
retriever = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX"),
    embedding=embeddings,
).as_retriever()

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--- Retrieving ---")
    question = state["question"]

    docs = retriever.invoke(question)

    # It is not necessary to pass question again as it is automatically carried over
    return {"question": question, "documents": docs}