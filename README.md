# ⚖️ Legal Document Agentic Assistant

> **🏆 CAPSTONE COMPLETION HIGHLIGHT**
> 
> I have successfully built this Capstone project covering **ALL 6 Mandatory Agentic Capabilities**:
> - ✅ **LangGraph StateGraph** (Complex 7-node flow)
> - ✅ **ChromaDB RAG** (10+ Domain Curated Documents)
> - ✅ **Conversation Memory** (Session-based thread caching)
> - ✅ **Self-Reflection** (LLM faithfulness-evaluating retry loop)
> - ✅ **Tool Use** (Integrated Python tool triggering mid-graph)
> - ✅ **Deployment** (Streamlit UI deployed efficiently)

An **Agentic AI** assistant built for Paralegals and Junior Lawyers to quickly answer questions from uploaded legal documents and standard contract clauses. This project was built utilizing LangGraph, ChromaDB, and Groq to streamline the process of reading and retrieving large volumes of case constraints.

---

## 🎯 Problem Statement

Paralegals and junior lawyers often spend considerable time manually reading through dense legal contracts to find specific clauses (such as Force Majeure, Severability, or Indemnification) and calculate applicable expiration timeframes.

This assistant provides **grounded, step-by-step guidance** utilizing a structured legal knowledge base (RAG), conversation memory, and a systemic faithfulness-measuring retry loop.

---

## ⚙️ The 6 Agentic Capabilities Built

| # | Capability | Implementation |
|---|-----------|---------------|
| 1 | **LangGraph StateGraph** | Complex workflow via nodes (`router`, `retrieve`, `tool`, `eval`, `answer`) tracking a global `CapstoneState`. |
| 2 | **ChromaDB RAG** | 10 highly-curated legal domains embedded locally using `SentenceTransformers`. |
| 3 | **Conversation Memory** | LangGraph `MemorySaver` keeps a sliding window history attached to an active `thread_id`. |
| 4 | **Self-Reflection (Eval)** | LLM-based faithfulness verifier. If the agent hallucinates outside the legal context (score < 0.7), it gets re-routed internally to try again. |
| 5 | **Tool Usage** | Implements a `current_date` retrieval python tool overriding RAG if the user requires current timestamp processing. |
| 6 | **Streamlit UI** | A completely ready-to-run responsive frontend, wrapping graph initializations inside `@st.cache_resource` for smooth rendering. |

---

## 📁 Repository Structure

```
AgenticAI_Project/
├── day13_capstone.ipynb      # The completed, structured Notebook demonstrating development iteratively
├── agent.py                  # The robust Agentic implementation (Loads DB, LLMs, and Graph Nodes)
├── capstone_streamlit.py     # The application UI deployment code
├── requirements.txt          # Python dependencies
├── .env                      # API keys (Add your GROQ key here!)
└── README.md                 # This file
```

---

## 📋 Knowledge Base Topics Simulated

1. Non-Disclosure Agreement (NDA)
2. Indemnification Clause
3. Force Majeure
4. Severability
5. Termination for Cause
6. Intellectual Property (IP) Assignment
7. Limitation of Liability
8. Dispute Resolution and Arbitration
9. Governing Law
10. Entire Agreement (Integration Clause)

---

## 🚀 Setup & Run Guide

### 1. Create a Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Your Groq Key
Create a `.env` file in the project's root folder and append your LLaMA API key:
```env
GROQ_API_KEY="gsk_your_api_key_goes_here"
```

### 4. Run the Streamlit Application
Start the Agentic frontend using Streamlit:
```bash
streamlit run capstone_streamlit.py
```

---

## 🧪 Testing the Capabilities

When the UI launches, test these edge cases to see the internal agent logic function:
- **Retrieval Test**: *“What does Severability mean in a contract?”*
- **Memory Test**: *“Can you summarize that last clause you explained?”*
- **Out of Scope Test**: *“Who won the last super bowl?”*
- **Tool Trigger**: *“What is the current time?”* 
- **Hallucination Red-Team**: *“Doesn't Force Majeure cover financial insolvency?”* *(Should forcefully reject based on the context constraints).*
