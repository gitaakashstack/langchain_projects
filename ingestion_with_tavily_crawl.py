import asyncio
import os
import ssl
import certifi
from langchain_classic import text_splitter
from pydantic import SecretStr
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
    output_dimensionality=768
)
vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ["PINECONE_INDEX"])

tavily_crawl = TavilyCrawl()

def create_chunks(results: List[Dict[str, Any]]) -> List[Document]:
    docs = [Document(page_content=result['raw_content'], metadata={"source": result['url'] }) for result in results]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

async def index_documents_async(docs: List[Document]):
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(docs)} documents to vector store",
        Colors.DARKCYAN,
    )
    try:
        await vectorstore.aadd_documents(docs)
        log_success(
            f"VectorStore Indexing: All documents processed successfully!"
        )
    except Exception as e:
        log_error(
            f"VectorStore Indexing: Failed to index documents - {e}"
        )

async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    # Crawl the documetnation site
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",
        "max_depth": 1,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })
    log_success(f"TavilyCrawl: Successfully crawled {len(res["results"])} URLs from documentation site.")

    all_docs = create_chunks(res["results"])

    log_success(f"Created {len(all_docs)} chunks from crawled content")

    await index_documents_async(all_docs)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents crawled: {len(res["results"])}")
    log_info(f"   • Chunks created: {len(all_docs)}")


if __name__ == "__main__":
    asyncio.run(main())