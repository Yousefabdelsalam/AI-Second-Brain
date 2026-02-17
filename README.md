# 🧠 AI Second Brain for PDFs (RAG Study Assistant)

Turn any PDF into a **personal AI tutor** using **RAG + LLMs**.  
This project helps students learn faster by generating explanations, summaries, quizzes, flashcards, study plans, and mind maps directly from PDF content.

---

## 🚀 Features

- ✅ **Explain Mode**: Clear explanations for any topic from the PDF  
- ✅ **Smart Summaries**: Short or detailed summaries  
- ✅ **Quiz Generator**: MCQs / short questions for self-testing  
- ✅ **Flashcards**: Q/A flashcards for memorization  
- ✅ **Study Plan**: Personalized study schedule based on your profile & exam date  
- ✅ **Topic Index**: Extract main topics & subtopics from the document  
- ✅ **Exam Style**: Generate exam-like questions from the PDF  
- ✅ **Weakness Analyzer**: Detect weak areas based on student answers  
- ✅ **Mind Map Tree**: Build a structured topic hierarchy (tree format)  
- ✅ **API Deployment**: FastAPI + Ngrok endpoints for external apps

---

## 🏗️ System Architecture

PDF → Text Extraction → Chunking → Embeddings → FAISS Vector DB  
→ Retriever → LLM → Study Mode Output

---

## 🧰 Tech Stack

- **Python**
- **LangChain**
- **HuggingFace Embeddings** (e.g., `sentence-transformers`)
- **FAISS** (Vector Database)
- **FastAPI** (API service)
- **Ngrok** (Public URL tunneling)
- *(Optional)* Streamlit for UI

---

## 📌 Project Modes

| Mode | What it does |
|------|--------------|
| `explain` | Explain a topic using retrieved PDF context |
| `summarize` | Summarize PDF content (topic-based) |
| `quiz` | Generate questions to test understanding |
| `flashcards` | Create flashcards for revision |
| `study_plan` | Build a study plan based on student profile |
| `topic_index` | Extract structured index of topics |
| `exam_style` | Generate exam-style questions |
| `weakness` | Analyze weaknesses from Q/A student answers |

---

AI-Second-Brain-for-PDFs/
│
├── AI_Second_Brain_for_PDFs.ipynb
├── requirements.txt
├── sample_pdfs/
│ └── example.pdf
├── assets/
│ └── architecture.png (optional)
└── README.md


---

## ⚙️ Installation

```bash
pip install -r requirements.txt
streamlit==1.32.2
python-dotenv==1.0.1
pypdf==4.0.2

langchain==0.2.16
langchain-core==0.2.38
langchain-community==0.2.16
langchain-text-splitters==0.2.4
langchain-groq==0.1.9

faiss-cpu==1.9.0.post1
sentence-transformers==3.0.1

networkx==3.2.1
matplotlib==3.8.4

▶️ Run (Notebook)

Open the notebook:

AI_Second_Brain_for_PDFs.ipynb

Run cells in order:

PDF extraction

chunking

embeddings + FAISS

LLM + prompts

API (FastAPI + Ngrok)

🌐 API Usage (FastAPI + Ngrok)
1) /generate

Generate content using a specific mode.


## 📂 Repository Structure (Suggested)

