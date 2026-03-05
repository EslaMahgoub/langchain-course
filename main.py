import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def main():
    print("Hello from langchain-course!")

LLM_CONFIG = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.environ.get("OPENROUTER_API_KEY"),
    "model": "gpt-5-nano",
    "temperature": 0,
}

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm = ChatOpenAI(**LLM_CONFIG)



if __name__ == "__main__":
    main()
