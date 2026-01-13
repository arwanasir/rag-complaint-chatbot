import faiss
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_vector_store(
    index_path="../vector_stores/complaints_index.faiss",
    metadata_path="../vector_stores/metadata.pkl"
):
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve_chunks(
    question,
    index,
    metadata,
    k=5
):
    # Embed question
    query_embedding = embedder.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    # FAISS search
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for idx in indices[0]:
        # ✅ pandas-safe row access
        row = metadata.iloc[int(idx)]
        retrieved_chunks.append(row.to_dict())

    return retrieved_chunks


def build_prompt(context_chunks, question):
    context_text = "\n\n".join(
        [f"- {chunk}" for chunk in context_chunks]
    )

    return f"""
You are a financial analyst assistant for CrediTrust.

Answer the user's question using ONLY the complaint excerpts provided below.
You may summarize, group, or synthesize recurring themes across multiple complaints
if the question asks for reasons, patterns, or common issues.

Do NOT introduce information that is not supported by the excerpts.
If the excerpts do not contain relevant information to answer the question,
say exactly:
"I don't have enough information from the complaints to answer this."

Complaint Excerpts:
{context_text}

Question:
{question}

Answer:
""".strip()


def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=200,
    )


def ask_question(question, k=5):
    index, metadata = load_vector_store()
    retrieved_chunks = retrieve_chunks(question, index, metadata, k)

    prompt = build_prompt(retrieved_chunks, question)

    llm = load_llm()
    response = llm(prompt)[0]["generated_text"]

    return {
        "answer": response,
        "sources": retrieved_chunks[:2]
    }


if __name__ == "__main__":
    question = "Why are customers unhappy with credit cards?"
    result = ask_question(question)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:\n")
    for src in result["sources"]:
        print("-", src["chunk"][:300])
