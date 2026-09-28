from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    response = web_search_tool.invoke({ 'query': question})
    results = response["results"]
    joined_tavily_result = "\n".join(
        [result["content"] for result in results]
    )
    web_results = Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return { "documents": documents }