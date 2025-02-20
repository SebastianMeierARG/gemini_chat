import os
import fitz  # PyMuPDF for PDF extraction
import google.generativeai as genai
import numpy as np
import faiss  # Vector search
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ✅ Initialize SentenceTransformer model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ✅ Load all PDFs from the folder
pdf_folder = "./pdf_files/"
documents = []
file_names = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file)
        text = extract_text_from_pdf(file_path)
        documents.append(text)
        file_names.append(file)  # Store file names for reference

# ✅ Convert all document texts into embeddings
doc_embeddings = embedder.encode(documents)
dimension = doc_embeddings.shape[1]

# ✅ Create FAISS index for fast retrieval
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# ✅ Retrieve relevant text from stored PDFs
def retrieve_relevant_text(query):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k=1)  # Top 1 match
    best_match = documents[indices[0][0]]  # Get most relevant text
    return best_match[:1000]  # Return only the first 1000 characters

# ✅ Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-pro")

def query_gemini(prompt):
    """Retrieve relevant text and generate a Gemini response."""
    retrieved_text = retrieve_relevant_text(prompt)  # Retrieve relevant info
    full_prompt = f"Context: {retrieved_text}\nUser question: {prompt}"
    
    response = model.generate_content(full_prompt)
    return response.text

# ✅ Example Query
response = query_gemini("Tell me about my professional experience.")
print(response)
