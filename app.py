# app.py
import streamlit as st
import PyPDF2
import pandas as pd
import os
from huggingface_hub import InferenceClient

# Load Hugging Face token from environment (set in Streamlit secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Setup inference client using flan-t5-large (free + public)
client = InferenceClient(model="google/flan-t5-large", token=HF_TOKEN)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def generate_flashcards(text, max_flashcards=5):
    short_text = text[:1000]
    prompt = (
        f"Generate {max_flashcards} flashcards from the following text.\n"
        f"Format each as:\nQ: question\nA: answer\n\n"
        f"Text:\n{short_text}"
    )
    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=512,
        temperature=0.7
    )
    return response

def parse_flashcards(output):
    lines = output.strip().split("\n")
    flashcards = []
    question = answer = ""
    for line in lines:
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question and answer:
                flashcards.append({"Question": question, "Answer": answer})
                question, answer = "", ""
    return pd.DataFrame(flashcards)

# --- Streamlit UI ---
st.set_page_config(page_title="Flashcard Generator", layout="centered")
st.title("🧠 LLM Flashcard Generator")

uploaded_file = st.file_uploader("📄 Upload a PDF textbook:", type=["pdf"])

if uploaded_file:
    st.success("File uploaded successfully!")
    pdf_text = extract_text_from_pdf(uploaded_file)

    if st.button("⚡ Generate Flashcards"):
        with st.spinner("Generating flashcards..."):
            raw_output = generate_flashcards(pdf_text)
            df = parse_flashcards(raw_output)

        st.success("✅ Flashcards generated!")
        st.dataframe(df)

        st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(), "flashcards.csv")
        st.download_button("⬇ Download JSON", df.to_json(orient="records", indent=2).encode(), "flashcards.json")
