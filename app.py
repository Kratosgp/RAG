from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
app = Flask(__name__)

def loading_dataKnowledge():
    # Load data from PDF and create vector representations
    loader = PyPDFLoader("case1/caseDoc.pdf")
    data = loader.load()
    documents = data
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    # Set up embeddings and create vector store
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    
    # Set up language model for question answering
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.7, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")
    
    return db, chain

def response(msg, db):
    # Define your response processing logic here
    docs = db.similarity_search(msg)
    return docs

@app.route('/api', methods=['GET'])
def get_bot_response():
    # Get the question from the request
    msg = request.args.get('question')
    
    # Load data and initialize question answering chain
    db, chain = loading_dataKnowledge()
    
    # Get response from the question answering chain
    bot_response = chain.run(input_documents=response(msg, db), question=msg)
    
    # Extract relevant information
    start_index = bot_response.find("Helpful Answer:")
    if start_index != -1:
        relevant_info = bot_response[start_index + len("Helpful Answer:"):].strip()
    else:
        relevant_info = "No relevant information found."
    
    return jsonify({'Answer': relevant_info})
if __name__ == '__main__':
    app.run(port=8080)
