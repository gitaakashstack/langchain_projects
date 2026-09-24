import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

load_dotenv()


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [
    UnstructuredLoader(
        web_url=url, chunking_strategy='basic', max_characters=1000000
        ).load()
        for url in urls
    ]
flattened_docs = [doc for inner_docs in docs for doc in inner_docs]

print(f"Created {len(flattened_docs)} documents from urls")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=0
)
chunks = text_splitter.split_documents(flattened_docs)

print(f"Created {len(chunks)} chunks after splitting")

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=os.getenv("OPENAI_API_KEY"),
)

print("Ingesting...")

PineconeVectorStore.from_documents(
    embedding=embeddings,
    documents=chunks,
    index_name=os.getenv("PINECONE_INDEX"),
)

print("Ingestion Finished")