import streamlit as st
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Legal Document Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ Legal Document Assistant")
st.caption("Answers questions based on standard legal clauses using Agentic AI.")

@st.cache_resource
def load_agent_and_docs():
    from agent import app, DOCUMENTS
    return app, DOCUMENTS

system_app, DOCUMENTS = load_agent_and_docs()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About")
    st.write("An Agentic Legal Assistant for Paralegals/Junior Lawyers.")
    st.write(f"Session Thread: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics Covered:**")
    topics = [d["topic"] for d in DOCUMENTS]
    for t in topics:
        st.write(f"• {t}")
        
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask a legal question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Processing legal documents..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = system_app.invoke({"question": prompt}, config=config)
                
                answer = result.get("answer", "I couldn't process that.")
                faith = result.get("faithfulness", 1.0)
                sources = result.get("sources", [])
                
                st.write(answer)
                if faith > 0:
                    st.caption(f"Faithfulness: {faith:.2f} | Sources: {sources}")
            except Exception as e:
                answer = f"Error processing request: {str(e)}"
                st.write(answer)
                
    st.session_state.messages.append({"role": "assistant", "content": answer})
