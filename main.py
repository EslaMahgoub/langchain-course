from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

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


llm = ChatOllama(model="qwen2.5", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)
parser = StrOutputParser()

def main():
    print("Hello from react-search-agent!")
    result = agent.invoke({"messages":  HumanMessage(content=("Search for 3 jobs postings for an ai engineer using python and langChain in Krakow"))})
    print(result)


if __name__ == "__main__":
    main()
