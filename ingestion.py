import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

LLM_CONFIG = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "gpt-5-nano",
        "temperature": 0,
    }

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=LLM_CONFIG["api_key"],
    openai_api_base=LLM_CONFIG["base_url"],
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10
)

#chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(index_name="langchain-rag-bot", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)


tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: list[Document], batch_size: int = 50):
    """ Process documents in batches asynchronously """
    log_header("VECTOR STORAGE PHASE")
    log_info(f"")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

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
    
    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

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
    """Main async function to orchestrate the entire process. """
    log_header("Documnetation ingestion Pipeline")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site", Colors.PURPLE,
    )

    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain",
        "max_depth": 5,
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })

    all_docs = res["results"]
    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result['url']}) for result in res['results']]
    log_success(
            f"TavilyCrawl: Successfully crawled {len(all_docs)} from documentation site"
        )
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
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


    openai_llm = ChatOpenAI(**LLM_CONFIG)
    parser = StrOutputParser()
    ollama = ChatOllama(temperature=0, model="qwen3:1.7b")


if __name__ == "__main__":
    asyncio.run(main())
