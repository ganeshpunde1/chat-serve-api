# RAG Pipeline Diagram Explanation

This diagram illustrates a **RAG (Retrieval-Augmented Generation)** pipeline — the process of preparing documents so an AI can search and retrieve relevant information from them. It has 4 stages:

---

## 1. 📥 LOAD — *Data Ingestion*

Raw documents from various sources are ingested. These can include:

- Code files, PDFs, text docs, spreadsheets
- Images, JSON files, URLs

All of them get collected into a unified document set ready for processing.

---

## 2. ✂️ SPLIT — *Chunks*

The loaded documents are **broken into smaller pieces** called "chunks." This is necessary because:

- LLMs have a limited context window
- Smaller chunks = more precise retrieval later

---

## 3. 🔢 EMBED — *Embeddings*

Each chunk is converted into a **numerical vector** (e.g., `[0.3, 0.4, 0.1, 1.8, 1.1...]`) by an embedding model. These numbers capture the **semantic meaning** of the text, so similar concepts have similar vectors.

---

## 4. 🗄️ STORE — *Vector DB*

The vectors are stored in a **Vector Database**, which enables **Document Similarity Search** — finding the chunks most semantically relevant to a user's query at retrieval time.

---

## Why It Matters

This pipeline is the foundation of how AI systems (like chatbots or Q&A tools) can answer questions grounded in **your own documents**, rather than relying solely on training data.

| Stage  | Action             | Output          |
|--------|--------------------|-----------------|
| Load   | Ingest documents   | Raw docs        |
| Split  | Chunk documents    | Text chunks     |
| Embed  | Vectorize chunks   | Number arrays   |
| Store  | Save to Vector DB  | Searchable index|
