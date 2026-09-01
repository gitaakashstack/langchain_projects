from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float) -> float:
    """
    Triples the input number
    Args:
        num: input number
    Returns:
        the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0).bind_tools(tools)