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

#chroma = Chroma(persist_directory="chroma_db", embedding_dunction=embeddings)
vectorstore = PineconeVectorStore(index_name="langchain-rag-bot", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def main():
    print("Hello")

    openai_llm = ChatOpenAI(**LLM_CONFIG)
    parser = StrOutputParser()
    ollama = ChatOllama(temperature=0, model="qwen3:1.7b")


if __name__ == "__main__":
    asyncio.run(main())
