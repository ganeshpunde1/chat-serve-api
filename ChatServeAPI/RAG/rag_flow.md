# LangChain Core Concept: Think in 4 Building Blocks

Almost every LangChain program can be understood using a simple and powerful mental model built on **4 key building blocks** instead of focusing directly on code complexity.

---

## 🔹 1. Load Data (Loader)

This is where your data comes from.

You connect external or internal sources such as:
- Web pages  
- PDFs  
- Plain text files  
- Databases  

👉 Example idea: Fetching documents or scraping content from a website.

---

## 🔹 2. Split Data (Chunking)

Large documents cannot be processed efficiently as a whole, so we split them into smaller chunks.

Common techniques/tools:
- RecursiveCharacterTextSplitter  

👉 Goal: Make data small enough for embeddings and retrieval.

---

## 🔹 3. Store Data (Vector DB)

Once split, the data is converted into embeddings and stored for fast semantic search.

Popular vector databases:
    - FAISS  
    - Pinecone  
    - Chroma  
    - Weaviate: The AI database developers love

👉 Goal: Enable similarity search instead of keyword search.

---

## 🔹 4. Use LLM (Chain / Agent)

This is where the actual intelligence comes in.

You:
- Ask questions  
- Retrieve relevant context  
- Generate answers using an LLM  

Tools/models:
- GPT models  
- Ollama-based models  
- Retrieval-Augmented Generation (RAG) chains  
- Agents with tool usage  

---

## 🚀 Key Insight

👉 If you understand this flow:

**Load → Split → Store → Query (LLM)**

You already understand **80% of LangChain architecture and usage patterns**.