from langgraph.graph import StateGraph, END

from graph.consts import RETRIEVE, GRADE_DOCUMENTS,GENERATE,WEBSEARCH
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.state import GraphState

def should_web_search(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE



graph = StateGraph(state_schema=GraphState)

graph.add_node(RETRIEVE, retrieve)
graph.add_node(GRADE_DOCUMENTS, grade_documents)
graph.add_node(GENERATE, generate)
graph.add_node(WEBSEARCH, web_search)

graph.set_entry_point(RETRIEVE)
graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
graph.add_edge(WEBSEARCH, GENERATE)
graph.add_edge(GENERATE, END)

graph.add_conditional_edges(
    GRADE_DOCUMENTS,
    should_web_search,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    })

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
