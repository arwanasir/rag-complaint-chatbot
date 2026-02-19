import re
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
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )


def ask_assistant(query, df_chunks, index, embed_model, model, tokenizer):
    # 1. Broaden the Search (Query Expansion)
    # We combine the semantic search with a keyword fallback to ensure context is found
    query_vector = embed_model.encode([query]).astype('float32')
    distances, sem_indices = index.search(
        query_vector, k=8)  # Increased k for more 'material'

    # Filter valid indices and get chunks
    retrieved_chunks = [df_chunks.iloc[i]['chunk']
                        for i in sem_indices[0] if i != -1]

    # 2. Keyword Fallback (Ensures context contains literal matches)
    keywords = [re.sub(r'\W+', '', word)
                for word in query.split() if len(word) > 3]
    for word in keywords:
        matches = df_chunks[df_chunks['chunk'].str.contains(
            word, case=False, na=False)]
        retrieved_chunks.extend(matches['chunk'].head(1).tolist())

    # Deduplicate and limit to fit the 512 token window
    combined_context = list(dict.fromkeys(retrieved_chunks))[:5]
    context_text = " ".join(combined_context)

    # 3. Balanced "Instruction-Tuning" Prompt
    # We move the 'safety' instruction to the end so it doesn't block the start of the answer.
    prompt = (
        f"Context: {context_text}\n\n"
        f"Question: {query}\n\n"
        f"Using the context provided above, write a brief, professional response. "
        f"If the information is absolutely missing, say I do not have enough information.\n"
        f"Answer:"
    )

    # 4. Final Generation Parameters (The 'Golden' Config)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=10,        # Prevents empty one-word answers
        repetition_penalty=1.2,   # Lowered to allow the model to use context keywords
        num_beams=5,              # Higher beam search for better sentence flow
        length_penalty=0.8,       # Slightly favors concise summaries
        early_stopping=True,
        do_sample=False
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean up any residual 'Answer:' tags
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    return response


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
