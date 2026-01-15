import streamlit as st
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="CrediTrust Advisor",
                   page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stApp { max-width: 1000px; margin: 0 auto; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    index = faiss.read_index("vector_stores/complaints_index.faiss")
    df = pd.read_pickle("vector_stores/metadata.pkl")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    gen_pipeline = pipeline("text2text-generation",
                            model=model, tokenizer=tokenizer)
    return df, index, embed_model, gen_pipeline


df_chunks, index, embed_model, generator = load_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.info("This assistant uses retrieved customer complaints to answer your financial queries.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about customer complaints..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching records..."):
            query_vec = embed_model.encode([prompt]).astype('float32')
            _, indices = index.search(query_vec, k=3)

            col_name = 'chunk' if 'chunk' in df_chunks.columns else df_chunks.columns[0]
            context = "\n\n".join(
                [str(df_chunks.iloc[i][col_name]) for i in indices[0]])

            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"

            result = generator(full_prompt, max_new_tokens=150)
            response = result[0]['generated_text']

            st.markdown(response)
            with st.expander(" View Data Sources"):
                st.write(context)

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
