from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import os

from pydantic import SecretStr

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="C:\\PersonalDrive\\LangChain Course\\Repos\\langchain-course\\mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000000
    )
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(document)
    print(f"create {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
        output_dimensionality=768
        )

    print("Ingesting...")

    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("PINECONE_INDEX"))

    print("Finished!")




