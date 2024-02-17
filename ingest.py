import os
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredURLLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "dburl")


# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Initialize loader
    urls = ['https://docs.gpt4all.io/', 'https://ollama.com/library/llama2']
    
    url_loader = UnstructuredURLLoader(urls = urls, show_progress_bar=True)
    loaded_documents = url_loader.load()
    #len(loaded_documents)

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(loaded_documents)
    #len(chunked_documents)
    #chunked_documents[0]

    # Initialize Ollama Embeddings
    ollama_embeddings = OllamaEmbeddings(model="mistral")

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=ollama_embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()
    
    # query it
    #query = "Who are the authors of the paper"
    #docs = vector_database.similarity_search(query)


    # print results
    #print(docs[0].page_content)


if __name__ == "__main__":
    create_vector_database()