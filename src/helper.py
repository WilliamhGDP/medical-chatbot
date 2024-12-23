from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

# Extract data from PDF
def load_pdf(data):
  loader = DirectoryLoader(data,
                  glob="*.pdf",
                  loader_cls=PyPDFLoader)
  documents = loader.load()

  return documents

# Split text into chunks
def text_split(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
  text_chunks = text_splitter.split_documents(extracted_data)

  return text_chunks

def download_hugging_face_embeddings():
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  return embeddings