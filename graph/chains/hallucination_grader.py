from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel

from graph.state import GraphState

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt taken from rlm/rag-hallucinations
system = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
    Give a binary score 1 or 0, where 1 means that the answer is grounded in / supported by the set of facts.
    """
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
