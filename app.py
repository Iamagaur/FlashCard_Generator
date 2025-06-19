import streamlit as st
import PyPDF2
import pandas as pd
import os
from huggingface_hub import InferenceClient

# 🔑 Replace this with your Hugging Face token
HF_TOKEN = "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"

# Load the Zephyr-7B model (works with Hugging Face Inference API)
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

# 📄 Extract text from uploaded PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# 🧠 Generate flashcards using LLM
def generate_flashcards(text, max_flashcards=8):
    short_text = text[:1000]
    prompt = (
        f"<|user|>\n"
        f"You are a tutor. Generate {max_flashcards} flashcards from this text:\n\n"
        f"{short_text}\n\n"
        f"Use this format only:\n"
        f"Q: ...\nA: ...\n\n"
        f"Return flashcards only.\n<|assistant|>\n"
    )

    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=512,
        temperature=0.7
    )

    return response

# 📊 Parse text into flashcard pairs
def parse_flashcards(text_output):
    lines = text_output.strip().split("\n")
    cards = []
    question = answer = None
    for line in lines:
        if "Q:" in line:
            question = line.split("Q:", 1)[1].strip()
        elif "A:" in line:
            answer = line.split("A:", 1)[1].strip()
            if question and answer:
                cards.append({"Question": question, "Answer": answer})
                question = answer = None
    return pd.DataFrame(cards)

# 🚀 Streamlit Web UI
st.set_page_config(page_title="Flashcard Generator", layout="centered")
st.title("📚 LLM Flashcard Generator")

uploaded_file = st.file_uploader("📁 Upload your textbook (PDF)", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded! Extracting content...")
    pdf_text = extract_text_from_pdf(uploaded_file)

    if st.button("⚡ Generate Flashcards"):
        with st.spinner("Thinking..."):
            output = generate_flashcards(pdf_text)
            df = parse_flashcards(output)

        st.success("✅ Flashcards generated successfully!")
        st.dataframe(df)

        # 📥 Download buttons
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "flashcards.csv", "text/csv")
        st.download_button("⬇️ Download JSON", df.to_json(orient="records", indent=2).encode(), "flashcards.json", "application/json")
