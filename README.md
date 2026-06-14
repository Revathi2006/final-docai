# 📄 DOC AI — Intelligent PDF Assistant

🚀 **Live Demo:**  
👉 https://huggingface.co/spaces/Revathi2006/document_ai_pdf


<img width="1860" height="851" alt="image" src="https://github.com/user-attachments/assets/90cd8ee3-673e-41c9-ba67-5053997740a2" />


---

**DOC AI** is a modern AI-powered web application that allows users to **upload PDFs**, **ask questions**, **generate summaries**, and **look up definitions** — all through a beautiful, animated UI and a powerful **FastAPI** backend.

It combines **semantic search**, **AI embeddings**, and **real-time document analysis** to turn static PDFs into interactive knowledge sources.

---

## ✨ Features

### 📤 Upload PDFs
- Drag & Drop or browse PDF files  
- Automatic text extraction  
- Smart text chunking for better search accuracy  

### ❓ Ask Questions
- Ask natural language questions from uploaded PDFs  
- Semantic search using vector embeddings  
- Returns most relevant answers with similarity score  

### 📚 Dictionary
- Instant definitions for any word  
- Uses:
  - Free Dictionary API  
  - Urban Dictionary (modern/slang terms)  
  - Merriam-Webster (fallback scraping)  
- Smart deduplication and clean formatting  

### 📝 PDF Summarization
- Generates concise summaries from entire documents  
- Local summarization (no paid APIs required)  

### 🎨 Modern UI
- Glassmorphism design  
- Smooth animations & transitions  
- Fully responsive (mobile + desktop)  
- Interactive modal views  

---

## 🧠 Tech Stack

### Frontend
- HTML5  
- CSS3 (Glassmorphism, animations)  
- Vanilla JavaScript  
- Font Awesome  
- Google Fonts (Poppins)  

### Backend
- FastAPI  
- Sentence Transformers (`all-MiniLM-L6-v2`)  
- ChromaDB (Vector Database)  
- pdfplumber  
- Uvicorn  
- Python  

---

## 🏗️ Architecture Overview

Frontend (HTML + JS)
|
| REST API Calls
v
FastAPI Backend
|
|-- PDF Processing (pdfplumber)
|-- Text Chunking
|-- Embedding Generation
|-- ChromaDB Vector Search
|-- Dictionary APIs

yaml
Copy code

---

## 🚀 Getting Started (Local Setup)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/doc-ai.git
cd doc-ai
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the Backend
bash
Copy code
uvicorn backend:app --reload
Open in browser:

cpp
Copy code
http://127.0.0.1:8000
🤗 Hugging Face Spaces Deployment
This project is successfully deployed and running on Hugging Face Spaces:

🔗 Live App:
https://huggingface.co/spaces/Revathi2006/document_ai_pdf

If you plan to deploy or customize the app on Hugging Face Spaces, you may need to define a YAML configuration block depending on the SDK (Docker / FastAPI).

📘 Configuration Reference:
https://huggingface.co/docs/hub/spaces-config-reference

⚠️ Important
Do NOT wrap normal README content inside --- unless you are intentionally defining Hugging Face YAML configuration.
Invalid YAML will cause build or parsing errors.

📡 API Endpoints
Method	Endpoint	Description
POST	/upload	Upload PDF
POST	/ask	Ask questions from PDFs
GET	/dictionary/{word}	Dictionary lookup
GET	/summarize/{file_id}	Summarize PDF
GET	/files	List uploaded PDFs
GET	/health	Health check

📂 Project Structure
arduino
Copy code
DOC-AI/
├── backend.py
├── chroma_db/
├── static/
├── templates/
│   └── index.html
├── requirements.txt
├── .env
└── README.md
🔐 Environment Variables
Create a .env file:

env
Copy code
HUGGINGFACE_API_KEY=your_key_here
(Optional — app works without paid APIs)

🧪 Example Use Cases
📘 Students asking questions from textbooks

📄 Research paper analysis

🧑‍💼 Resume & policy document review

🧠 Learning assistant for PDFs

📚 Terminology lookup

⚠️ Limitations
Scanned PDFs without text won’t work (OCR not included)

Local summarization (not GPT-based)

Best suited for English documents

🌟 Future Enhancements
OCR support (Tesseract)

Multi-PDF cross-document querying

User authentication

Docker & Cloud deployment

GPT-based summarization

👨‍💻 Author
Revathi S
B.Tech AI & Data Science 
Saveetha Engineering College

