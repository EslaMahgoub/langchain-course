import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

def main():
    loader = TextLoader("/home/eslam/ai_projects/react-search-agent/mediumblog1.txt", encoding="UTF-8")
    document = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Document splited into: {len(texts)}")

    LLM_CONFIG = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "gpt-5-nano",
        "temperature": 0,
        # Embedding model (OpenRouter/OpenAI-compatible; chat model above is for LLM only)
        "embedding_model": "openai/text-embedding-3-small",
    }

    print(f"Embedding...")
    embeddings = OpenAIEmbeddings(
        model=LLM_CONFIG["embedding_model"],
        openai_api_key=LLM_CONFIG["api_key"],
        openai_api_base=LLM_CONFIG["base_url"],
    )

    print(f"Ingestion...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
                           
    print(f"Done...")
    # openai_llm = ChatOpenAI(**LLM_CONFIG)
    # ollama = ChatOllama(temperature=0, model="gemma3:4b")

    # chain = summary_prompt_template | ollama | parser
    # response = chain.invoke(input={"information": information})


if __name__ == "__main__":
    main()
