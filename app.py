from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Pinecone API key and environment
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_HOST = os.environ.get("PINECONE_API_HOST")
INDEX_NAME = os.environ.get("INDEX_NAME")

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME, host=PINECONE_API_HOST)

# Initialize Pinecone vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Prompt template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# Initialize the LLM transformer
llm=CTransformers(model="./model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={"max_new_tokens":512,
                          "temperature":0.8})

# Initialize the chain
qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)