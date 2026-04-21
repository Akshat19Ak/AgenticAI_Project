import os
import chromadb
from datetime import datetime
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from sentence_transformers import SentenceTransformer

load_dotenv()

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Non-Disclosure Agreement (NDA)",
        "text": "A Non-Disclosure Agreement (NDA) legally binds parties to keep certain information confidential. It typically specifies what information is protected, the duration of the obligation, and any exceptions (such as information already in the public domain). Breaching an NDA can lead to legal action, including injunctions and damages."
    },
    {
        "id": "doc_002",
        "topic": "Indemnification Clause",
        "text": "An indemnification clause requires one party to compensate the other for certain damages or losses arising from the contract. This often applies to third-party claims. For example, a software vendor may indemnify a client against copyright infringement claims related to the software's use."
    },
    {
        "id": "doc_003",
        "topic": "Force Majeure",
        "text": "Force majeure clauses excuse one or both parties from performing contractual obligations due to unforeseen, unavoidable events (e.g., acts of God, pandemics, war). If invoked successfully, it prevents the delayed or non-performed obligations from being considered a breach of contract."
    },
    {
        "id": "doc_004",
        "topic": "Severability",
        "text": "The severability clause ensures that if one part of a contract is found to be invalid or unenforceable by a court, the remainder of the contract remains in effect. This prevents the entire agreement from becoming void due to a single problematic section."
    },
    {
        "id": "doc_005",
        "topic": "Termination for Cause",
        "text": "Termination for cause allows a party to end a contract if the other party breaches its material obligations. The clause typically requires giving written notice and providing a cure period (e.g., 30 days) allowing the breaching party to fix the issue before termination is finalized."
    },
    {
        "id": "doc_006",
        "topic": "Intellectual Property (IP) Assignment",
        "text": "IP assignment clauses dictate who owns the intellectual property created during the term of a contract. In employment or contractor agreements, it typically states that any inventions, software, or designs created by the worker belong exclusively to the employer or client."
    },
    {
        "id": "doc_007",
        "topic": "Limitation of Liability",
        "text": "A Limitation of Liability clause caps the amount of damages one party can recover from another in the event of a breach. It often excludes indirect, incidental, or consequential damages entirely, and limits direct damages to the total amount paid under the contract."
    },
    {
        "id": "doc_008",
        "topic": "Dispute Resolution and Arbitration",
        "text": "Dispute resolution clauses outline how disagreements should be resolved. Many business contracts mandate binding arbitration instead of litigation, requiring parties to present their case to an arbitrator. This is generally faster and more private than going to court."
    },
    {
        "id": "doc_009",
        "topic": "Governing Law",
        "text": "The governing law clause specifies which state or country's laws will apply to the interpretation and enforcement of the contract. This is crucial for multi-state or international agreements, as the legal standards for breach and damages can vary significantly between jurisdictions."
    },
    {
        "id": "doc_010",
        "topic": "Entire Agreement (Integration Clause)",
        "text": "The entire agreement clause, also known as an integration clause, states that the written contract represents the complete and final agreement between the parties. It supersedes any prior oral or written negotiations, meaning promises made outside the written contract cannot be enforced."
    }
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

client = chromadb.Client()
try:
    client.delete_collection("legal_kb")
except:
    pass
collection = client.create_collection("legal_kb")

texts = [d["text"] for d in DOCUMENTS]
ids   = [d["id"]   for d in DOCUMENTS]
embeddings = embedder.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
)

class CapstoneState(TypedDict):
    question:      str
    messages:      List[dict]
    route:         str
    retrieved:     str
    sources:       List[str]
    tool_result:   str
    answer:        str
    faithfulness:  float
    eval_retries:  int

def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:  
        msgs = msgs[-6:]
    return {"messages": msgs}

def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

    prompt = f"""You are a router for a chatbot assisting Paralegals with Legal Documents.

Available options:
- retrieve: search the knowledge base for topics like NDA, Terminations, Indemnification, Severability.
- memory_only: answer from conversation history (e.g. 'what did you just say?', or conversational filler).
- tool: use the current_date tool. Use this ONLY if the user asks for the current date, time, or what day it is today.

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    if "memory" in decision:       decision = "memory_only"
    elif "tool" in decision:       decision = "tool"
    else:                          decision = "retrieve"

    return {"route": decision}

def retrieval_node(state: CapstoneState) -> dict:
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}

def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}

def tool_node(state: CapstoneState) -> dict:
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tool_result = f"System Current Date and Time: {current_date}"
    return {"tool_result": tool_result}

def answer_node(state: CapstoneState) -> dict:
    question    = state["question"]
    retrieved   = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages    = state.get("messages", [])
    eval_retries= state.get("eval_retries", 0)

    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result:
        context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    if context:
        system_content = f"""You are a Legal Document Assistant for Paralegals.
Answer using ONLY the legal context or tool output provided below.
If the answer is not in the context, say: I don't have that information in the legal knowledge base.
Do NOT use outside knowledge to give legal advice.

{context}"""
    else:
        system_content = "You are a helpful Legal Document Assistant. Answer based on the conversation history."

    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Your previous answer contained unverified claims. Answer using ONLY information explicitly stated in the context above."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

def eval_node(state: CapstoneState) -> dict:
    answer   = state.get("answer", "")
    context  = state.get("retrieved", "")[:500]
    retries  = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5
    return {"faithfulness": score, "eval_retries": retries + 1}

def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}

def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"

def eval_decision(state: CapstoneState) -> str:
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"

graph = StateGraph(CapstoneState)
graph.add_node("memory",    memory_node)
graph.add_node("router",    router_node)
graph.add_node("retrieve",  retrieval_node)
graph.add_node("skip",      skip_retrieval_node)
graph.add_node("tool",      tool_node)
graph.add_node("answer",    answer_node)
graph.add_node("eval",      eval_node)
graph.add_node("save",      save_node)

graph.set_entry_point("memory")
graph.add_edge("memory",   "router")
graph.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
graph.add_edge("retrieve", "answer")
graph.add_edge("skip",     "answer")
graph.add_edge("tool",     "answer")
graph.add_edge("answer", "eval")
graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
graph.add_edge("save", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

def invoke_agent(question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return {
        "answer": result.get("answer", "I couldn't process that."),
        "sources": result.get("sources", []),
        "faithfulness": result.get("faithfulness", 1.0)
    }
