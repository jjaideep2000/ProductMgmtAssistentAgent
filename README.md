# 🤖 Product Management Assistant – AI Agent

A modular AI-powered assistant that supports product managers by answering queries related to **features**, **competitive intelligence**, and **customer insights**. It uses a smart multi-agent architecture backed by **Amazon Bedrock**, **OpenSearch**, and **serverless Lambda functions**.

---

## 🧠 What It Does

- 🔍 **Classifies** user queries (e.g., "Give me a new feature idea" → Feature Agent).
- 🧠 **Retrieves relevant documents** using vector similarity from OpenSearch.
- 💬 **Generates high-quality responses** using LLMs (LLaMA 3 via Bedrock).
- 🧭 **Routes intelligently** to the right agent: Feature, Insight, or Competitive.

---

## 📦 Technologies Used

| Technology         | Purpose                                         |
|--------------------|-------------------------------------------------|
| **Amazon Bedrock** | LLM inference (LLaMA 3 + Titan Embeddings)      |
| **OpenSearch**     | Vector similarity search                        |
| **AWS Lambda**     | Docker-based backend for inference and routing  |
| **Amazon S3**      | Stores embedded documents                       |
| **LangChain**      | Retrieval-Augmented Generation (RAG)            |
