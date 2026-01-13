"""
RAG Pipeline for Complaint Analysis
Follows Task 3 instructions exactly
"""

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')


class ComplaintRAG:

    def __init__(self, vector_store_path="../vector_stores/", embedding_model_name="all-MiniLM-L6-v2"):
        print(" Loading RAG system...")

        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f" Loaded embedding model: {embedding_model_name}")

        self.index = faiss.read_index(
            f"{vector_store_path}/complaints_index.faiss")
        with open(f"{vector_store_path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        print(f"Loaded FAISS index with {self.index.ntotal} chunks")

        # 4. Initialize the LLM (using a small, free model)
        print("🤖 Loading language model...")
        self.llm = pipeline(
            "text-generation",
            model="google/flan-t5-large",
            torch_dtype="auto",
            device_map="auto"
        )
        print(" Language model ready")

        self.prompt_template = """You are a financial analyst assistant for CreditTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

**Context**: {context}

**Question**: {question}

**Answer**: """

    def retrieve(self, question, top_k=5):

        question_embedding = self.embedding_model.encode([question])

        distances, indices = self.index.search(question_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata['chunk']):
                result = {
                    'text': self.metadata['chunk'][idx],
                    'distance': float(distance),
                    'metadata': {k: self.metadata[k][idx] for k in self.metadata if k != 'chunk'}
                }
                results.append(result)

        return results

    def format_context(self, retrieved_chunks):
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks[:3]):
            context_parts.append(f"[Source {i+1}] {chunk['text']}")

        return "\n\n".join(context_parts)

    def generate_answer(self, question, retrieved_chunks):

        context = self.format_context(retrieved_chunks)
        prompt = self.prompt_template.format(
            context=context, question=question)

        response = self.llm(
            prompt,
            max_length=300,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.llm.tokenizer.eos_token_id
        )

        full_text = response[0]['generated_text']
        if "**Answer**: " in full_text:
            answer = full_text.split("**Answer**: ")[1].strip()
        else:
            answer = full_text

        return answer

    def ask_question(self, question, top_k=5):
        print(f"\nQuestion: {question}")
        print(" Retrieving relevant complaints...")

        sources = self.retrieve(question, top_k=top_k)

        print(f"Retrieved {len(sources)} relevant complaint chunks")

        print(" Generating answer...")
        answer = self.generate_answer(question, sources)

        return answer, sources

    def evaluate(self, test_questions):
        results = []

        for i, (question, expected) in enumerate(test_questions):
            print(f"\n🔬 Evaluating Question {i+1}: {question}")

            answer, sources = self.ask_question(question)

            quality_score = 3
            if "don't have enough information" in answer.lower():
                quality_score = 1
            elif len(answer.split()) > 10:
                quality_score = 4

            result = {
                'Question': question,
                'Generated Answer': answer[:200] + "..." if len(answer) > 200 else answer,
                'Retrieved Sources': [s['metadata'].get('product_category', 'Unknown') for s in sources[:2]],
                'Quality Score': quality_score,
                'Comments': self._analyze_answer(question, answer, sources)
            }

            results.append(result)

            print(f" Score: {quality_score}/5")

        return pd.DataFrame(results)

    def _analyze_answer(self, question, answer, sources):
        if not sources:
            return "No relevant sources found"

        if "credit card" in question.lower() and any("credit card" in str(s['metadata'].get('product_category', '')).lower() for s in sources):
            return "Good: Retrieved credit card complaints"

        if len(answer.split()) < 10:
            return "Answer might be too short"

        return "Answer seems reasonable based on context"


def load_prebuilt_vector_store(base_path="data/"):

    return None


if __name__ == "__main__":
    rag = ComplaintRAG()

    test_question = "Why are people unhappy with Credit Cards?"
    answer, sources = rag.ask_question(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")
    print(f"\nSources used: {len(sources)} complaint chunks")
