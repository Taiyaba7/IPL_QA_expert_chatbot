import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re

# --- Gemini API Setup ---
genai.configure(api_key="AIzaSyBl6LID0pQoVPie0g9HGtrYnedLOKugZDo")

# --- Load PDF Q&A Dataset ---
pdf_path = r"C:\Users\SMILE\Desktop\zenith\data\IPL_QA_Dataset_1000.pdf"#
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

# Extract Q and A pairs
questions, answers = [], []
q_matches = re.findall(r"Q\d+:\s*(.*)", text)
a_matches = re.findall(r"A\d+:\s*(.*)", text)

if not q_matches or not a_matches:
    st.error("⚠️ No Q&A found in the PDF. Please check formatting.")
else:
    questions = q_matches
    answers = a_matches

# --- Create Embeddings ---
model = SentenceTransformer("all-MiniLM-L6-v2")
question_embeddings = model.encode(questions, show_progress_bar=True)

# --- FAISS Index ---
dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(question_embeddings))

# --- Streamlit UI ---
st.title("IPL Question Answering Bot 🏏 (PDF-based)")
user_q = st.text_input("Ask your IPL question:")

if user_q:
    # Step 1: Find most similar Q
    q_embed = model.encode([user_q])
    D, I = index.search(np.array(q_embed), k=3)  # top 3 matches
    
    context = "\n".join([f"Q: {questions[i]}\nA: {answers[i]}" for i in I[0]])
    
    # Step 2: Use Gemini with context
    prompt = f"""
    You are an IPL assistant. 
    Use the following knowledge base to answer accurately:
    
    {context}
    
    User Question: {user_q}
    Answer:
    """
    
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    st.write(response.text)
