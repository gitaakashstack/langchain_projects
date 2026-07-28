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

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i: i+chunk_size])
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs"""
    try:
        log_info(f"TavilyExtract: Starting to extract batch {batch_num} with {len(urls)} URLs", Colors.BLUE)
        docs = await tavily_extract.ainvoke({ "urls": urls })
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}", Colors.RED)
        return []

async def async_extract(url_batches: List[List[str]]):
    pass


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(f"TavilyMap: Successfully mapped {len(site_map["results"])} URLs from documentation site")

    # Splitting urls into batches of 20
    url_batches = chunk_urls(site_map["results"], 20)
    log_info(f"URL Processing: Split {len(site_map["results"])} URLs into {len(url_batches)} batches", Colors.BLUE)

if __name__ == "__main__":
    asyncio.run(main())