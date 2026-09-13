from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

from chains import generation_chain, reflection_chain



class MessageGraph(TypedDict):
    messages: Annotated[list[HumanMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflect_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6 :
        return END
    return REFLECT

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflect_node)
builder.set_entry_point(GENERATE)
builder.add_conditional_edges(GENERATE, should_continue, path_map={
    END: END,
    REFLECT: REFLECT,
})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path='flow.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hello React LangGraph with Reflection')
    res = graph.invoke({
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                        @LangChainAI
                — newly Tool Calling feature is seriously underrated.

                After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

                Made a video covering their newest blog post

                                      """
            )
        ]
    })
    print(res)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
