#  Gemini RAG Chatbot: Personal AI Chatbot Using Your Documents

This project is an **AI-powered chatbot** that uses **Google's Gemini model** combined with **Retrieval-Augmented Generation (RAG)** to answer questions based on your personal documents (e.g., **CV, recommendation letters, cover letters**).

---

##  Features

-  **Retrieves information from PDFs** (CV, cover letter, etc.)
-  **Uses FAISS for fast similarity search**
-  **Improved retrieval with reranking (cosine similarity)**
-  **Provides document sources for transparency**
-  **Interactive web UI using Gradio**
-  **Deployable locally or in the cloud**

---

##  Folder Structure

```
📁 gemini_chatbot/
 ├── 📁 pdf_files/      # Place your PDFs here (CV, recommendation, etc.)
 ├── finetuned_app.py   # Main Python script
 ├── requirements.txt   # Required dependencies
 ├── .env               # Store your Gemini API Key
 ├── README.md          # This file
```

---

##  Requirements

### ** Install Dependencies**

Ensure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

### ** API Key Setup**

1. Get an **API Key** from [Google AI Studio](https://ai.google.dev/)
2. Create a `.env` file and add:

```env
GEMINI_API_KEY=your_api_key_here
```

---

##  How It Works

1. **Extracts text** from PDFs in `pdf_files/`
2. **Generates embeddings** using `SentenceTransformers`
3. **Stores embeddings in FAISS** for fast retrieval
4. **Retrieves the most relevant document sections** for a user query
5. **Sends retrieved text as context** to Gemini for a personalized response
6. **Displays results in a Gradio Web UI**

---

##  Run the Application

```bash
python finetuned_app.py
```

This will launch a **web interface** where you can chat with the AI.

---

##  Example Queries

You can ask questions like:

- "Tell me about my professional experience."
- "What are my strongest skills?"
- "Summarize my recommendation letter."

---

## Gradio Web UI

The chatbot is hosted in a **Gradio interface**, which allows for easy interactions.

- **User Input:** Ask a question
- **AI Response:** The chatbot generates a response
- **Sources:** Shows which documents were used



