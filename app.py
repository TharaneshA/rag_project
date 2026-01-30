import streamlit as st
import requests
import time

BACKEND = "http://localhost:8000/api"

st.set_page_config(page_title="RAG + Structured Data Demo", layout="wide")
st.title("Excel Q&A Chatbot")

with st.sidebar:
    st.header("Upload Data")
    file = st.file_uploader("Choose Excel/CSV file", type=["xlsx", "csv"])
    
    if file:
        if st.button("Process File", type="primary"):
            try:
                resp = requests.post(
                    f"{BACKEND}/upload/",
                    files={"file": (file.name, file.getvalue())},
                    timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Uploaded {data['rows']} rows!")
                    st.session_state['file_uploaded'] = True
                else:
                    st.error(resp.json().get('error', 'Failed'))
            except Exception as e:
                st.error(str(e))
    
    # Status with auto-refresh
    if st.session_state.get('file_uploaded'):
        placeholder = st.empty()
        try:
            status = requests.get(f"{BACKEND}/status/", timeout=5).json()
            
            if status['processing']:
                progress = status['progress'] / max(status['total'], 1)
                placeholder.progress(progress, f"{status.get('message', 'Processing...')} ({status['progress']}/{status['total']})")
                time.sleep(0.5)
                st.rerun()
            elif status['error']:
                placeholder.error(f"Error: {status['error'][:200]}")
            elif status['ready']:
                placeholder.success("Ready! Ask questions below.")
        except:
            pass

st.header("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "method" in msg:
            st.caption(f"Method: {msg['method']}")

if query := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{BACKEND}/chat/", json={"query": query}, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.write(data["answer"])
                    st.caption(f"Method: {data['method']}")
                    st.session_state.messages.append({
                        "role": "assistant", "content": data["answer"], "method": data["method"]
                    })
                else:
                    st.error(resp.json().get("error", "Failed"))
            except Exception as e:
                st.error(str(e))
