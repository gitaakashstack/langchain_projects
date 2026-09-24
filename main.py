from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
RESPONDER = "draft"
REVISOR = "revise"
TOOL = "execute_tools"


def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal[TOOL, END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return TOOL

builder = StateGraph(MessagesState)
builder.add_node(RESPONDER, draft_node)
builder.add_node(TOOL, execute_tools)
builder.add_node(REVISOR, revise_node)

builder.add_edge(START, RESPONDER)
builder.add_edge(RESPONDER, TOOL)
builder.add_edge(TOOL, REVISOR)

builder.add_conditional_edges(REVISOR, event_loop, path_map={
    END: END,
    TOOL: TOOL,
})

app = builder.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

res = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)
# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)

