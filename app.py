import streamlit as st
import PyPDF2
import pandas as pd
import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token not found. Set it in Streamlit secrets.")
    st.stop()

client = InferenceClient(model="mrm8488/t5-base-finetuned-question-generation-ap", token=HF_TOKEN)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def generate_flashcards(text, max_flashcards=5):
    short_text = text[:1000]
    prompt = (
        f"Generate {max_flashcards} flashcards from the following text.\n\n"
        f"Text:\n{short_text}\n\n"
        f"Format:\nQ: ...\nA: ..."
    )
    response = client.text_generation(prompt=prompt, max_new_tokens=512, temperature=0.7)
    return response

def parse_flashcards(text_output):
    lines = text_output.strip().split("\n")
    cards = []
    question = answer = None
    for line in lines:
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
            if question and answer:
                cards.append({"Question": question, "Answer": answer})
                question = answer = None
    return pd.DataFrame(cards)

st.set_page_config(page_title="Flashcard Generator", layout="centered")
st.title("📚 LLM Flashcard Generator")

uploaded_file = st.file_uploader("📁 Upload your textbook (PDF)", type=["pdf"])

if uploaded_file:
    st.success("✅ File uploaded")
    text = extract_text_from_pdf(uploaded_file)

    if st.button("⚡ Generate Flashcards"):
        with st.spinner("Generating..."):
            output = generate_flashcards(text)
            df = parse_flashcards(output)
        st.success("✅ Flashcards generated!")
        st.dataframe(df)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "flashcards.csv")
        st.download_button("⬇️ Download JSON", df.to_json(orient="records", indent=2).encode(), "flashcards.json")