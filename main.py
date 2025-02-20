import os
import fitz  # PyMuPDF for PDF extraction
import google.generativeai as genai
import numpy as np
import faiss  # Vector search
import gradio as gr
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize SentenceTransformer for Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Load all PDFs from the folder
pdf_folder = "./pdf_files/"
documents = []
file_names = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, file)
        text = extract_text_from_pdf(file_path)
        documents.append(text)
        file_names.append(file)  # Store file names for reference

# Convert all document texts into embeddings
doc_embeddings = embedder.encode(documents)
dimension = doc_embeddings.shape[1]

# Create FAISS index for fast retrieval
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Retrieve top 3 most relevant texts
def retrieve_relevant_texts(query, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k=top_k)  # Retrieve top_k matches
    
    retrieved_texts = []
    sources = []
    
    for i in range(top_k):
        retrieved_texts.append(documents[indices[0][i]][:1000])  # First 1000 chars per document
        sources.append(file_names[indices[0][i]])  # Corresponding file name
    
    return "\n\n".join(retrieved_texts), sources  # Join multiple retrieved texts

# Initialize Gemini Model
model = genai.GenerativeModel("gemini-1.5-pro")

def query_gemini(prompt):
    """Retrieve relevant documents and generate a response using Gemini."""
    retrieved_text, sources = retrieve_relevant_texts(prompt)  # Retrieve multiple texts
    full_prompt = f"Context: {retrieved_text}\nUser question: {prompt}"
    
    response = model.generate_content(full_prompt)
    return response.text, sources

#  Gradio Chat UI
def chatbot_interface(user_input, history=[]):
    """
    Handles chat interaction with Gemini.
    - user_input: User's message
    - history: Chat history
    """
    response, sources = query_gemini(user_input)
    history.append((user_input, response))  # Update chat history
    return history, f"Sources: {', '.join(sources)}"




if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## 🤖 Chat with Your AI, Trained on Your Documents")
        
        chatbot = gr.Chatbot(label="Gemini AI Chatbot")
        msg = gr.Textbox(label="Your message:", placeholder="Ask about your CV, skills, or experience...")
        submit_btn = gr.Button("Send")
        sources_text = gr.Textbox(label="Retrieved Sources", interactive=False)

        # Functionality
        submit_btn.click(chatbot_interface, inputs=[msg, chatbot], outputs=[chatbot, sources_text])
    demo.launch()