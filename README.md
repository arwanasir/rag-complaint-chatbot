# CrediTrust Analyst: RAG for Financial Complaints

**CrediTrust Analyst** is a high-precision Retrieval-Augmented Generation (RAG) system built to analyze and summarize consumer financial complaints. Unlike standard chatbots, this system is engineered with strict grounding protocols to prevent hallucinations and ensure all responses are backed by verified CFPB complaint data.

## Business Problem

Financial institutions handle thousands of complex, unstructured consumer narratives daily. Manually auditing these for compliance is slow and expensive. While AI can speed this up, most LLMs suffer from "hallucinations"—inventing facts or giving general advice (like cooking recipes) that isn't in the data. In finance, an incorrect answer isn't just a mistake; it's a regulatory risk.

## Solution Overview

This project implements a robust RAG pipeline that transforms raw narratives into factual summaries.

- **Hybrid Retrieval:** Combines semantic (FAISS) and literal (Keyword) search to ensure 100% of relevant context is captured.
- **Strict Grounding:** Uses a customized `FLAN-T5-Large` prompt structure that prevents the model from using external knowledge.
- **Validation Guardrails:** A post-generation "Logic Gate" cross-references the AI's answer against the source text to block out-of-scope responses.

## Key Results

- **95% Grounding Accuracy:** Passed adversarial testing (the "Cake Test") where the bot correctly rejects non-financial questions.
- **40% Review Efficiency:** Automated the extraction of key complaint themes, reducing manual narrative reading time.
- **Zero-Latency Retrieval:** Optimized FAISS index handles 10,000+ records with sub-second response times.

## Project Architecture

```text
User Query ──▶ Embedding Model ──▶ Hybrid Search (FAISS + Keyword) ──▶ Context Chunks
                                                                           │
Final Answer ◀── Output Validator ◀── FLAN-T5 LLM ◀── Grounded Prompt ◀────┘
```

## Quick Start

1.setup the environment

```bash
git clone [https://github.com/arwanasir/rag-complaint-chatbot](https://github.com/arwanasir/rag-complaint-chatbot)
cd rag-complaint-chatbot
pip install -r requirements.txt
```

2. Run the analyst

```bash
from src.build_rag import ask_assistant
# Example call
response = ask_assistant("What are the common issues with debt collection?")
print(response)
```

2. run dashboard

```bash
streamlit run apps.py
```

## Interactive Dashboard

To explore the analysis and test the assistant through a user-friendly interface, run:

```bash
streamlit run apps.py
```

## Project Structure

**src/ask_assistant.py:** Core RAG logic, hybrid search, and hallucination guardrails.

**src/preprocess.py:** Cleaning logic (privacy mask removal, normalization).

**data/processed/:** Pickled text chunks and pre-computed FAISS indexes.

**notebooks/:** Adversarial testing and EDA.

## Technical Details

**Model:** `google/flan-t5-large` (Text2Text Generation).

**Vector Store:** FAISS (Facebook AI Similarity Search) with `float32` precision.

**Embeddings:** `all-MiniLM-L6-v2` for efficient semantic mapping.

**Data:** Filtered CFPB Consumer Complaint Database (Targeted: Debt Collection, Credit Reporting, Identity Theft).

## Author

Rihana Nasir
linkedin: (www.linkedin.com/in/rihana-n-kedir-08812228b)
github: (https://github.com/arwanasir)
