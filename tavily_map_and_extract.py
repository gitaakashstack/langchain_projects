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
from google import genai

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

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunks.append(urls[i: i+chunk_size])
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract content from a batch of URLs"""
    try:
        log_info(f"TavilyExtract: Starting to extract batch {batch_num} with {len(urls)} URLs", Colors.BLUE)
        docs = await tavily_extract.ainvoke({ "urls": urls })
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}", Colors.RED)
        return []

async def async_extract(url_batches: List[List[str]]):
    """Converts the content extracted from """
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"🔧 TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, i) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:  # type: ignore
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]},
                )
                all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")

    return all_pages

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Index documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    client = genai.Client()
    total_tokens = sum(client.models.count_tokens(
        model="gemini-embedding-001",
        contents=doc.page_content,
    ).total_tokens for doc in documents[0:batch_size])
    print(f"Total tokens in a batch: {total_tokens}")

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Processing 3 batches concurrently in 1 minute
    results = []
    for i, batch in enumerate(batches):
        await add_batch(batch, i)
        log_info(
            "🗺️  Sleeping for 1 minute to allow rate limits to reset",
            Colors.PURPLE,
        )
        await asyncio.sleep(60)
        log_info(
            "🗺️  1 minute over ",
            Colors.PURPLE,
        )

    # Process batches concurrently
    # tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    # results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        "🗺️  TavilyMap: Starting to map documentation structure from \"https://python.langchain.com\" ",
        Colors.PURPLE,
    )

    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(f"TavilyMap: Successfully mapped {len(site_map["results"])} URLs from documentation site")

    # Splitting urls into batches of 20
    url_batches = chunk_urls(site_map["results"], 20)
    log_info(f"URL Processing: Split {len(site_map["results"])} URLs into {len(url_batches)} batches", Colors.BLUE)

    all_docs = await async_extract(url_batches)

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=25)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • URLs mapped: {len(site_map['results'])}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())