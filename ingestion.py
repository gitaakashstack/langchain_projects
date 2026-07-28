import asyncio
import os
import ssl
import certifi
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

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

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
    all_docs = res["results"]
    log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} URLs from documentation site.")

if __name__ == "__main__":
    asyncio.run(main())