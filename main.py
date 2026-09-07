from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import  TavilyClient

load_dotenv()
tavily = TavilyClient()

@tool
def search_weather(query: str) -> dict[str, Any]:
   """
    Tool that searches over the internet
    Args:
        query: The query to search over
    Returns:
        The search result
   """
   print(f"Searching for {query}")
   return tavily.search(query)


llm = ChatOllama(temperature=0, model="qwen3.5:0.8b")
tools = [search_weather]
agent = create_agent(model=llm, tools=tools)



def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage("What is the weather in Tokyo, Delhi and Berlin?")})
    print(result)
if __name__ == "__main__":
    main()
