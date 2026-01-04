# rag-complaint-chatbot

# Consumer Complaint Analysis & Preprocessing

## 📌 Project Overview

This project focuses on building a data pipeline for the CFPB (Consumer Financial Protection Bureau) dataset. We have filtered, profiled, and cleaned thousands of consumer narratives to prepare them for a RAG-based (Retrieval-Augmented Generation) chatbot.

## 📁 Project Structure

- `data/raw/`: Original, unaltered dataset.
- `data/processed/`: Filtered and cleaned versions of the data.
- `notebooks/`: Exploratory Data Analysis (EDA) and cleaning scripts.
- `.gitignore`: Configured to exclude large CSV files and environment folders.

## 🛠️ Tasks Completed

### 1. Data Filtering & Profiling

- **Targeted Scope**: Filtered the dataset to include only the 5 specific products requested for analysis.
- **Null Handling**: Removed all rows with missing "Consumer complaint narratives" to ensure quality text input.
- **Metrics**: Generated word count statistics and visualized the distribution of narrative lengths.

### 2. Narrative Cleaning (Preprocessing)

- **Normalization**: Converted all text to lowercase for consistency.
- **Noise Removal**: Stripped out special characters (e.g., `>`, `!`, `?`) and removed the `xxxx` privacy masks.
- **Tokenization Readiness**: Preserved word spacing to ensure the text is readable by both humans and AI models.

## 🚀 Getting Started

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```
