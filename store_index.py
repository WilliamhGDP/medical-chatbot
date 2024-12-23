from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv
from uuid import uuid4
import os

# Load environment variables
load_dotenv()

# Pinecone API key and environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_HOST = os.environ.get("PINECONE_API_HOST")

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Load data
data_folder_path = "./data"
extracted_data = load_pdf(data_folder_path)

# Split text into chunks
text_chunks = text_split(extracted_data)

# Initialize Pinecone
index_name = os.environ.get("INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name, host=PINECONE_API_HOST)

# Initialize Pinecone vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Store index
documents = [Document(page_content=t.page_content) for t in text_chunks]
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
