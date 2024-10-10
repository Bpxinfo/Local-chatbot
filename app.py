import os
import logging
import time
import multiprocessing
from functools import partial
import torch
from gpt4all import GPT4All
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import socket
from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
# model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Block internet access by overriding socket connect
def block_internet():
    def guard(*args, **kwargs):
        raise RuntimeError("Internet access is blocked for this script.")
    socket.socket.connect = guard

block_internet()

# Load documents (only PDFs and Text Files for now)
def load_document(file_path):
    try:
        if file_path.endswith('.pdf'):
            logging.info(f"Loading PDF document: {file_path}")
            loader = PyMuPDFLoader(file_path)
            return loader.load()
        elif file_path.endswith('.txt'):
            logging.info(f"Loading Text document: {file_path}")
            loader = TextLoader(file_path)
            return loader.load()
        else:
            logging.warning(f"Unsupported file format: {file_path}")
            return []
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {str(e)}")
        return []

def load_documents_from_folder(folder_path):
    documents = []
    # Use multiprocessing to load documents in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        file_paths = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files]
        results = pool.map(load_document, file_paths)
        for result in results:
            documents.extend(result)
    return documents

# Split documents into manageable chunks
def split_documents(documents):
    logging.info("Splitting documents into chunks...")
    total_length = sum(len(doc.page_content) for doc in documents)
    avg_length = total_length // len(documents) if documents else 512

    # Adjust chunk size and overlap dynamically based on document length
    chunk_size = min(max(512, avg_length // 2), 2048)
    chunk_overlap = min(chunk_size // 4, 256)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(documents)

# Generate embeddings for document chunks
def generate_embeddings(chunks, tokenizer, model):
    logging.info("Generating embeddings for document chunks...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return texts, np.array(embeddings)

# Initialize Transformers Model
def initialize_embedding_model(model_name="nomic-embed-text-v1.5"):
    logging.info(f"Initializing Transformers model: {model_name}")
    model_path = os.path.join(os.getcwd(), "models", model_name)
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_safetensors=True)
    else:
        raise FileNotFoundError(f"Model file does not exist: {model_path}. Please download it locally before running.")
    return tokenizer, model

# Initialize GPT4All Model
def initialize_model(model_name):
    logging.info(f"Initializing GPT4All model: {model_name}")
    # Set model path explicitly to avoid defaulting to the cache directory
    model_path = os.path.join(os.getcwd(), "models", model_name)  # Absolute path to the models folder
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    logging.info(f"Using model at path: {model_path}")
    return GPT4All(model_name=model_path, device='cpu', allow_download=False)

# Handle Chat Loop
def chat_loop(model, texts, embeddings, tokenizer, embedding_model):
    logging.info("Starting chat loop...")
    with model.chat_session():
        while True:
            query = input("User: ")
            if query.lower() in ["exit", "quit"]:
                logging.info("Exiting chat loop...")
                break
            start = time.time()
            # Perform similarity search manually using cosine similarity
            inputs = tokenizer(query, return_tensors='pt')
            with torch.no_grad():
                query_embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            similarity_scores = cosine_similarity([query_embedding], embeddings)[0]
            top_k_indices = np.argsort(similarity_scores)[-5:][::-1]
            results = [texts[i] for i in top_k_indices]
            context = "\n".join(results)

            response = model.generate(f"Based on the following context, answer the question: \n{context}\nQuestion: {query}", max_tokens=512)
            print(f"Chatbot: {response}")
            end = time.time()
            logging.info(f"Query processed in {end - start:.2f} seconds")

# Test Cases
def run_tests():
    logging.info("Running tests...")
    # Load sample documents for testing
    sample_folder = "sample_documents"
    documents = load_documents_from_folder(sample_folder)
    if not documents:
        logging.error("No documents loaded during test")
        return
    chunks = split_documents(documents)
    if not chunks:
        logging.error("Document splitting failed during test")
        return
    tokenizer, embedding_model = initialize_embedding_model()
    texts, embeddings = generate_embeddings(chunks, tokenizer, embedding_model)
    if not embeddings.any():
        logging.error("Embeddings generation failed during test")
        return
    model = initialize_model("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    if not model:
        logging.error("Model initialization failed during test")
        return
    logging.info("All tests completed successfully")

if __name__ == "__main__":
    # Specify path to folder containing documents to be loaded
    folder_path = "documents_folder"
    llama = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    documents = load_documents_from_folder(folder_path)
    chunks = split_documents(documents)
    tokenizer, embedding_model = initialize_embedding_model()
    texts, embeddings = generate_embeddings(chunks, tokenizer, embedding_model)
    model = initialize_model(llama)
    chat_loop(model, texts, embeddings, tokenizer, embedding_model)
    # Run tests (optional)
    run_tests()

