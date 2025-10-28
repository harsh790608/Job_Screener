import os
import time
import asyncio
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
import docx
from io import BytesIO
from dotenv import load_dotenv
import faiss
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import Tool

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- ENV --------------------
load_dotenv()
st.set_page_config(page_title="Agentic Screener + FAISS Async + Evaluation", layout="wide")
st.title("🧠 Agentic Resume Screener")

# -------------------- ENV VARS --------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo")
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

# -------------------- Helper Functions --------------------
def extract_text(uploaded_file):
    """
    Extract text content from uploaded files of type PDF, DOCX, or TXT.

    Args:
        uploaded_file (UploadedFile): File object uploaded via Streamlit.

    Returns:
        str: Extracted text content from the file.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = "\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )
            logger.info(f"Extracted text from PDF: {uploaded_file.name}")
            return text
        elif name.endswith(".docx"):
            document = docx.Document(BytesIO(uploaded_file.read()))
            text = "\n".join(
                p.text for p in document.paragraphs if p.text
            )
            logger.info(f"Extracted text from DOCX: {uploaded_file.name}")
            return text
        elif name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            logger.info(f"Extracted text from TXT: {uploaded_file.name}")
            return text
        else:
            logger.warning(f"Unsupported file format: {uploaded_file.name}")
    except Exception as e:
        st.warning(f"Failed to extract {uploaded_file.name}: {e}")
        logger.exception(f"Error extracting text from {uploaded_file.name}: {e}")
    return ""

# -------------------- Sidebar --------------------
st.sidebar.header("Azure / Run Settings")
job_description = st.text_area("Paste Job Description (required)", height=200)
files = st.file_uploader("Upload resumes (.pdf, .docx, .txt)", accept_multiple_files=True)
run = st.button("Run Screening")

# -------------------- Main Execution --------------------
if run:
    if not job_description.strip():
        st.error("Please provide a Job Description.")
        st.stop()
    if not files:
        st.error("Please upload at least one resume file.")
        st.stop()

    start_time = time.time()
    logger.info("Screening started.")

    with st.spinner("Initializing LLMs..."):
        try:
            llm = AzureChatOpenAI(
                openai_api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_API_VERSION,
                model_name=AZURE_OPENAI_MODEL_NAME,
                temperature=0.3
            )
            embeddings = AzureOpenAIEmbeddings(
                deployment=AZURE_EMBEDDING_DEPLOYMENT,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY
            )
            logger.info("LLM and embeddings initialized successfully.")
        except Exception as e:
            st.error(f"Failed to initialize LLMs: {e}")
            logger.exception("Error initializing LLM or embeddings.")
            st.stop()

    # -------------------- Async LLM Functions --------------------
    async def summarize_resume(resume_text):
        """Summarize resume into key points."""
        msgs = [
            SystemMessage(content="You are an expert AI recruiter analyzing technical resumes."),
            HumanMessage(content=f"Summarize this resume into key points: skills, experience, certifications, and 1-line professional summary.\n\n{resume_text}")
        ]
        try:
            response = await llm.ainvoke(msgs)
            return response.content
        except Exception as e:
            logger.exception("Error summarizing resume.")
            return f"Error summarizing resume: {e}"

    async def compare_with_job(summary, job_description):
        """Compare resume summary with job description."""
        msgs = [
            SystemMessage(content="You are an AI recruiter assessing candidate-job fit."),
            HumanMessage(content=f"First give the verdict whether it is good fit or poor fit or better suitable verdict followed by the summary with the job description and provide a 2-3 sentence justification.\n\nJob Description:\n{job_description}\n\nSummary:\n{summary}")
        ]
        try:
            response = await llm.ainvoke(msgs)
            return response.content
        except Exception as e:
            logger.exception("Error comparing job description.")
            return f"Error comparing with job: {e}"

    async def suggest_improvements(summary, job_description):
        """Suggest improvements to align resume with job description."""
        msgs = [
            SystemMessage(content="You are an AI career coach."),
            HumanMessage(content=f"Suggest exactly 3 actionable improvements to align this resume with the job description.\n\nJob Description:\n{job_description}\n\nSummary:\n{summary}\n\nOutput as bullet points.")
        ]
        try:
            response = await llm.ainvoke(msgs)
            return response.content
        except Exception as e:
            logger.exception("Error suggesting improvements.")
            return f"Error suggesting improvements: {e}"

    # -------------------- Tools --------------------
    summarize_tool = Tool(name="SummarizerTool", func=summarize_resume, description="Summarizes resumes.")
    compare_tool = Tool(name="FitEvaluatorTool", func=lambda text: compare_with_job(text, job_description), description="Evaluates job fit.")
    improve_tool = Tool(name="ImprovementTool", func=lambda text: suggest_improvements(text, job_description), description="Suggests improvements.")

    # -------------------- Process Resumes Async --------------------
    async def process_resume(file):
        """
        Process each resume: extract text, summarize, compare, and suggest improvements.
        """
        resume_text = extract_text(file)
        if not resume_text:
            return {"Candidate": file.name, "Summary": "No text", "Justification": "", "Suggestions": "", "Embedding": np.zeros(1536)}

        try:
            summary = await summarize_tool.func(resume_text)
            justification = await compare_tool.func(summary)
            suggestions = await improve_tool.func(summary)
            res_emb = np.array(embeddings.embed_query(summary), dtype=np.float32)
            logger.info(f"Processed {file.name} successfully.")
            return {"Candidate": file.name, "Summary": summary, "Justification": justification, "Suggestions": suggestions, "Embedding": res_emb}
        except Exception as e:
            logger.exception(f"Error processing {file.name}.")
            return {"Candidate": file.name, "Summary": f"Error: {e}", "Justification": "", "Suggestions": "", "Embedding": np.zeros(1536)}

    async def run_all():
        """Run all resume evaluations asynchronously."""
        try:
            return await asyncio.gather(*(process_resume(f) for f in files))
        except Exception as e:
            logger.exception("Async execution error.")
            st.error(f"Error during parallel processing: {e}")
            return []

    with st.spinner("🤖 Running agentic evaluation..."):
        results_list = asyncio.run(run_all())

    if not results_list:
        st.error("No results generated. Check logs for details.")
        st.stop()

    # -------------------- FAISS Ranking --------------------
    try:
        embeddings_list = [r["Embedding"] for r in results_list]
        candidate_names = [r["Candidate"] for r in results_list]
        dim = len(embeddings_list[0])
        index = faiss.IndexFlatIP(dim)
        emb_matrix = np.vstack(embeddings_list)
        faiss.normalize_L2(emb_matrix)
        index.add(emb_matrix)

        jd_emb = np.array(embeddings.embed_query(job_description), dtype=np.float32)
        faiss.normalize_L2(jd_emb.reshape(1, -1))
        D, I = index.search(jd_emb.reshape(1, -1), len(files))
        logger.info("FAISS similarity search completed.")
    except Exception as e:
        st.error(f"FAISS ranking failed: {e}")
        logger.exception("FAISS ranking error.")
        st.stop()

    # -------------------- Results --------------------
    final_results = []
    for rank, idx in enumerate(I[0]):
        r = results_list[idx]
        score = round(float(D[0][rank]) * 100, 2)
        final_results.append({
            "Candidate": r["Candidate"],
            "Similarity %": score,
            "Rank": rank + 1,
            "Summary": r["Summary"],
            "Justification": r["Justification"],
            "Suggestions": r["Suggestions"]
        })

    df = pd.DataFrame(final_results)
    st.success("✅ Screening Complete!")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "agentic_results.csv", "text/csv")

    # -------------------- EVALUATION METRICS --------------------
    st.header("📊 Evaluation Metrics")

    ai_scores = [r["Similarity %"] / 100 for r in final_results]
    ai_ranks = [r["Rank"] for r in final_results]
    n = len(ai_scores)

    avg_latency = (time.time() - start_time) / len(files)
    stability = (1 - (np.std(ai_scores) / np.mean(ai_scores))) * 100
    K = min(3, len(files))
    precision_at_k = sum(ai_ranks[i] <= K for i in range(K)) / K * 100
    recall = (sum(ai_ranks[i] <= K for i in range(K)) / K) * 100

    metrics = {
        "Precision@K (%)": round(precision_at_k, 2),
        "Recall (%)": round(recall, 2),
        "Stability (%)": round(stability, 2),
        "Average Latency (s/resume)": round(avg_latency, 2),
    }

    st.write(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
    st.info("✅ Metrics calculated: Precision@K, Recall, Stability (%), and Average Latency.")
    logger.info("Screening process completed successfully.")
