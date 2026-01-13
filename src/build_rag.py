from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import torch
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def load_resources(vector_path, pkl_path):
    df_chunks = pd.read_pickle(pkl_path)
    index = faiss.read_index(vector_path)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return df_chunks, index, embed_model


def get_llm_pipeline():
    model_id = "google/flan-t5-large"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def ask_assistant(query, df_chunks, index, embed_model, generator):
    query_vector = embed_model.encode([query]).astype('float32')
    _, indices = index.search(query_vector, k=5)
    context = "\n\n".join([df_chunks.iloc[i]['chunk'] for i in indices[0]])

    prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer 
                questions about customer complaints. Use the following retrieved complaint 
                excerpts to formulate your answer. If the context doesn't contain the 
                answer, say "i don't have enough information." 
    Context: {context}
    Question: {query}
    Answer:"""

    result = generator(prompt, max_new_tokens=150)
    return result[0]['generated_text']


"""def ask_assistant(query, df_chunks, index, embed_model, generator):
    query_vector = embed_model.encode([query]).astype('float32')
    _, indices = index.search(query_vector, k=5)
    context = "\n\n".join([df_chunks.iloc[i]['chunk']for i in indices[0]])
    # prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    prompt = You are a financial analyst assistant for CrediTrust. Your task is to answer 
                questions about customer complaints. Use the following retrieved complaint 
                excerpts to formulate your answer. If the context doesn't contain the 
                answer, say "i don't have enough information." 
    'Context: {context}
    Question :{query}
    Answer:

    result = generator(prompt, max_new_tokens=150, num_beams=5,
                       repetition_penalty=2.5, no_repeat_ngram_size=3, early_stopping=True)
    return result[0]['generated_text']
    """
