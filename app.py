import streamlit as st
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

st.set_page_config(page_title="CrediTrust Assistant", layout="wide")
st.title("CrediTrust Financial Analyst Assistant")


@st.cache_resource
def load_system():
    index = faiss.read_index("../vector_stores/complaints_index.faiss")
    df = pd.read_pickle("../vector_stores/metadata.pkl")
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


try:
    df_chunks, index, embed_model, generator = load_system()
    st.sidebar.success("System Ready")
except Exception as e:
    st.error(f" Error: {e}")
    st.info(
        "Check if 'complaint_index.faiss' and 'complaint_index.pkl' are in this folder.")
    st.stop()

query = st.text_input("Ask a question about customer complaints:")

if query:
    with st.spinner("Searching documents..."):
        query_vec = embed_model.encode([query]).astype('float32')
        distances, indices = index.search(query_vec, k=3)
        col_name = 'chunk' if 'chunk' in df_chunks.columns else df_chunks.columns[0]
        context = "\n\n".join([str(df_chunks.iloc[i][col_name])
                              for i in indices[0]])

        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer 
questions about customer complaints. Use the following retrieved complaint 
excerpts to formulate your answer. If the context doesn't contain the 
answer, say "i don't have enough information." 

Context: {context}

Question: {query}

Answer:"""

        result = generator(prompt, max_new_tokens=150)
        answer = result[0]['generated_text']

        st.markdown(" Assistant Response")
        st.write(answer)

        with st.expander("Show retrieved excerpts (Evidence)"):
            st.write(context)
