# 🧠 Agentic Resume Screener

An interactive **Streamlit** web application that automates resume screening using **Azure OpenAI**, **LangChain**, and **FAISS**.  
It evaluates candidate resumes against a given job description, generates summaries, fit justifications, and improvement suggestions — all asynchronously for faster performance.

---

## 🚀 Features

- 📝 **Resume Parsing** — Extracts text from `.pdf`, `.docx`, and `.txt` files.
- 🤖 **LLM-Powered Screening** — Uses Azure OpenAI’s GPT model for:
  - Resume summarization
  - Fit evaluation (Good/Poor/Better suitable)
  - Resume improvement suggestions
- ⚡ **Async Processing** — Screens multiple resumes in parallel using `asyncio`.
- 🧮 **Similarity Ranking (FAISS)** — Ranks candidates based on semantic similarity with the job description.
- 📊 **Evaluation Metrics** — Computes metrics like Precision@K, Recall, Stability, and Average Latency.
- 📥 **Downloadable Results** — Exports all results to a CSV file.

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | Python 3.10+ |
| LLM | Azure OpenAI (ChatGPT & Embeddings) |
| Framework | LangChain |
| Vector Index | FAISS |
| Async | asyncio |
| Logging | Python logging module |
| Document Parsing | PyPDF2, python-docx |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/agentic-resume-screener.git
cd agentic-resume-screener
