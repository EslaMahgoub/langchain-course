import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)


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


#Initialize vector store

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(index_name="langchain-rag-bot", embedding=embeddings)

# Initialize chat model using OpenRouter config
# Uses the model and temperature from LLM_CONFIG, and relies on OPENROUTER_API_KEY from the environment.
# model = init_chat_model(
#     f"openrouter:{LLM_CONFIG['model']}",
#     temperature=LLM_CONFIG["temperature"],
# )

# model = init_chat_model("qwen3:1.7b", model_provider="ollama")

model = ChatOpenAI(**LLM_CONFIG)
# ollama = ChatOllama(temperature=0, model="qwen3:1.7b")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # Build messages List
    # messages = [{"role": "user", "content": query}]
    messages = HumanMessage(content=query)

    #Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)
    