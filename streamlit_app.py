import streamlit as st
import requests
import uuid

# Configuration
API_URL = "http://localhost:7860"  # Match your FastAPI port

st.set_page_config(page_title="Tigra AI Support", page_icon="🤖")

# 1. Initialize Session State for Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    # Generate a persistent thread_id for the session to maintain LangGraph memory
    st.session_state.thread_id = str(uuid.uuid4())

st.title("🤖 Tigra AI Support Assistant")

# 2. Sidebar: Health Check & Metadata
with st.sidebar:
    st.header("System Status")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            st.success("API Online")
        else:
            st.error("API Issues Detected")
    except:
        st.error("Cannot connect to API")

    st.info(f"Session ID: {st.session_state.thread_id}")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 3. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message and message["metadata"].get("escalated"):
            st.caption("⚠️ This request has been flagged for human review.")

# 4. User Input Handling
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call your FastAPI /chat endpoint
    with st.chat_message("assistant"):
        with st.spinner("Tigra is thinking..."):
            try:
                payload = {
                    "user_input": prompt,
                    "thread_id": st.session_state.thread_id
                }
                response = requests.post(f"{API_URL}/chat", json=payload)
                response.raise_for_status()
                data = response.json()

                full_response = data["response"]
                metadata = data.get("metadata", {})

                # Display response
                st.markdown(full_response)

                # Append to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "metadata": metadata
                })

                if metadata.get("escalated"):
                    st.warning("Request escalated to human support.")

            except Exception as e:
                st.error(f"Error communicating with backend: {e}")
