from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss


def stratified_sample(df):
    sample_size = 15000
    fraction = sample_size/len(df)
    df_sample, _ = train_test_split(
        df,
        train_size=fraction,
        stratify=df['Product'],
        random_state=42
    )
    df_sample = pd.DataFrame(df_sample)
    print(f"sample created with {len(df_sample)} rows")
    return df_sample


def text_split(df_sample):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]

    )

    def chunk_narratives(row):
        text = row['cleaned_narratives']
        chunks = text_splitter.split_text(text)
        return [{"chunk": c, "Product": row['Product'], "complaint_id": row.name} for c in chunks]
    all_chunks = []
    for _, row in df_sample.iterrows():
        all_chunks.extend(chunk_narratives(row))
    df_chunks = pd.DataFrame(all_chunks)
    print(f"Original narratives: {len(df_sample)}")
    print(f"Total chunks created: {len(df_chunks)}")
    print(df_chunks.head())

    return df_chunks


def embed_and_index(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings... please wait.")
    embeddings = model.encode(df['chunk'].tolist(), show_progress_bar=True)

    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Index created with {index.ntotal} vectors.")
    faiss.write_index(index, "../vector_stores/complaints_index.faiss")
    df.to_pickle("../vector_stores/metadata.pkl")
    print("Vector store and metadata saved successfully!")
    return df
