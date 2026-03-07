import re

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

from schemas import AgentResponse

load_dotenv()

tavily = TavilyClient()

import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": OPENROUTER_API_KEY,
    "model": "gpt-5-nano",
    "temperature": 0,
}

openai_llm = ChatOpenAI(**LLM_CONFIG)

# #Manual tavily search tool
# @tool
# def search(query: str) -> str:
#     """ 
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

#Using langchain-tavily package

llm = ChatOllama(model="qwen2.5:latest", temperature=0)
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    response_format=AgentResponse,
    tools=tools
)

def main():
    print("Hello from react-search-agent!")
    prompt = (
        "Search for exactly 2 job postings for an AI engineer using Python and LangChain in Krakow on LinkedIn. "
        "Do at most one search, then return your answer with 2 job postings and their sources."
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt)]}  # cap steps so it doesn't keep searching
    )
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
