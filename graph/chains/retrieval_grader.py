from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_message = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
human_message = "Retrieved document: \n\n {document} \n\n User question: {question}"

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", human_message),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader