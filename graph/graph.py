from langgraph.graph import StateGraph, END

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import RETRIEVE, GRADE_DOCUMENTS,GENERATE,WEBSEARCH
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search
from graph.state import GraphState

def should_web_search(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({ "documents": documents, "generation": generation })

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
    else:
        return RETRIEVE

graph = StateGraph(state_schema=GraphState)

graph.add_node(RETRIEVE, retrieve)
graph.add_node(GRADE_DOCUMENTS, grade_documents)
graph.add_node(GENERATE, generate)
graph.add_node(WEBSEARCH, web_search)

graph.add_edge(RETRIEVE, GRADE_DOCUMENTS)
graph.add_edge(WEBSEARCH, GENERATE)
graph.add_edge(GENERATE, END)

graph.set_conditional_entry_point(route_question, path_map={
    WEBSEARCH: WEBSEARCH,
    RETRIEVE: RETRIEVE,
})

graph.add_conditional_edges(
    GRADE_DOCUMENTS,
    should_web_search,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    })

graph.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "useful": END,
        "not useful": WEBSEARCH,
        "not supported": GENERATE,
    }
)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
