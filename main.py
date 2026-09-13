from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()

from chains import generation_chain, reflection_chain




if __name__ == '__main__':
    print('Hello React LangGraph with Reflection')
   

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
