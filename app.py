# app.py
import streamlit as st
import os
import sys
import io
import traceback
from typing import List, Dict, Any

# Keep project root on path so utils and models import works when running from this file.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import project modules.
from models.llm import get_chatgroq_model
from utils.rag_utils import retrieve, chunk_text, index_documents
from utils.web_search import serpapi_search

# --- Helper functions -------------------------------------------------------------------

def build_context_for_query(query, top_k=4):
    """Get top-k docs from Chroma and return a single context string to pass to the LLM."""
    try:
        docs = retrieve(query, k=top_k)
    except Exception as e:
        return ""

    if not docs:
        return ""  # no docs

    # Filter out low-similarity docs
    relevant_docs = [d for d in docs if d.get("distance", 100) < 1.5]
    if not relevant_docs:
        return ""  # fallback will trigger

    pieces = []
    for i, d in enumerate(relevant_docs):
        pieces.append(f"Source {i+1} (score {d.get('distance',0):.4f}): {d['text']}")
    return "\n\n---\n\n".join(pieces)


def get_chat_response(chat_model, messages: List[Dict[str, str]], system_prompt: str, retrieval_context: str, response_mode: str = "detailed") -> str:
    """Compose messages and call the GROQ chat model with optional retrieval context and response-mode instructions."""
    try:
        formatted = [SystemMessage(content=system_prompt)]
        # Inject retrieval context as additional system message if present.
        if retrieval_context:
            formatted.append(SystemMessage(content=f"Relevant context from documents:\n{retrieval_context}"))
        # Append conversation history.
        for msg in messages:
            if msg.get("role") == "user":
                formatted.append(HumanMessage(content=msg.get("content", "")))
            else:
                formatted.append(AIMessage(content=msg.get("content", "")))
        # Response mode hints.
        if response_mode.lower() == "concise":
            formatted.append(SystemMessage(content="Be concise: 2-3 sentences max, direct answers only, no examples or background."))
        else:
            formatted.append(SystemMessage(content="Be detailed: 5+ sentences with examples, step-by-step reasoning, context, and best practices."))
        # Call model.
        response = chat_model.invoke(formatted)
        # Model response object assumed to have '.content'.
        return getattr(response, "content", str(response))
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error getting response: {str(e)}.\n\nTraceback:\n{tb}"

# --- Pages -----------------------------------------------------------------------------

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation and Requirements
    1. Create and activate a virtual environment.
    2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    3. Required libraries include `streamlit`, `chromadb`, `sentence-transformers`, `langchain_groq`, `python-dotenv`.
    """)

    st.markdown("""
    ## API Key Setup
    Create a `.env` file in the project root or set environment variables in your OS.
    Example `.env` entries:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    GROQ_MODEL=llama-3.1-8b-instant
    SERPAPI_KEY=your_serpapi_key_here
    CHROMA_PERSIST_DIR=./chroma_db
    EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    ```
    """)

    st.markdown("""
    ## 📝 Available Models

    ### Groq Models
    Popular models:
    - `llama-3.1-70b-versatile` - Large, powerful
    - `llama-3.1-8b-instant` - Fast, smaller
    - `mixtral-8x7b-32768` - Balanced speed/capability
    """)

    st.markdown("""
    ## How RAG Works
    - Upload documents on the Chat page to index them into ChromaDB using sentence-transformers embeddings.
    - When you ask a question, the app retrieves the top K document chunks and sends them as context to the Groq model.
    - If no relevant documents are found, a web-search fallback is used to provide context snippets.
    """)

    st.markdown("""
    ## How to Use
    1. Go to the **Chat** page using the sidebar.
    2. Select **response mode**: `Detailed` or `Concise`.
    3. Optionally, set a **System Prompt** to customize the AI's personality or behavior.
    4. Upload documents to index into ChromaDB.
    5. Type your question and get answers using either local documents or web search.
    """)

    st.markdown("""
    ## Tips
    - **System Prompts**: Customize the AI’s personality and behavior.
    - **Response Modes**: `Detailed` gives long, in-depth answers. `Concise` gives short, direct answers.
    - **Chat History**: Persists during your session but resets on refresh.
    """)

    st.markdown("""
    ## Troubleshooting
    - **API Key Issues**: Ensure your GROQ or SERPAPI keys are valid.
    - **Model Not Found**: Check the `GROQ_MODEL` variable matches available Groq models.
    - **Connection Errors**: Verify your internet connection and API service status.
    """)

    st.markdown("---")
    st.markdown("Ready to start chatting? Navigate to the **Chat** page using the sidebar!")
  

def chat_page():
    """Main chat interface page with RAG and indexing capabilities."""
    st.title("👾 AI ChatBot")

    # Basic session_state defaults.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Detailed"
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""

    # Top-level controls visible on the chat page.
    st.write("Use the sidebar to clear chat, upload documents, and choose response mode.")
    st.divider()

    # Try to initialize the Groq chat model.
    model_available = True
    try:
        chat_model = get_chatgroq_model()
    except Exception as e:
        chat_model = None
        model_available = False
        st.warning("Model initialization failed or API key not found. Chat with the model is disabled until a valid GROQ API key is configured.")
        st.caption(f"Error: {str(e)}")

    # Display chat history.
    for message in st.session_state.messages:
        # st.chat_message accepts 'user' or 'assistant'.
        role = message.get("role", "user")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))

    # Chat input is shown regardless, but it will error if model not available.
    prompt = st.chat_input("Type your message here...")

    # If user submitted a message via chat_input.
    if prompt:
        # Add user message to chat history.
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Immediately render user's message.
        with st.chat_message("user"):
            st.markdown(prompt)

        # If the model is not available, show info and skip model call.
        if not model_available:
            assistant_text = "Model not initialized. Please set GROQ_API_KEY in your environment and restart the app."
            with st.chat_message("assistant"):
                st.markdown(assistant_text)
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            st.stop()

        # Build retrieval context.
        retrieval_context = build_context_for_query(prompt, top_k=4)
        used_fallback_search = False

        # If no docs found, attempt fallback web search snippets.
        if not retrieval_context:
            snippets = serpapi_search(prompt, num_results=3)
            if snippets:
                retrieval_context = "\n\n".join(snippets)
                used_fallback_search = True

        # Prepare system prompt and response mode.
        system_prompt = st.session_state.get("system_prompt", "")
        response_mode = st.session_state.get("response_mode", "Detailed").lower()

        # Call the model and show a spinner.
        # Call the model and show a spinner.
        with st.chat_message("assistant"):
            with st.spinner("Getting response from model..."):
                try:
                    response_text = get_chat_response(
                        chat_model,
                        st.session_state.messages,
                        system_prompt,
                        retrieval_context,
                        response_mode
                    )
                except Exception as e:
                    response_text = f"Error while generating response: {str(e)}"

                # Add mode + fallback tags
                mode_label = "📝 Concise Mode" if response_mode == "concise" else "📖 Detailed Mode"
                if used_fallback_search:
                    mode_label += " + 🌍 Web Search"

                # Show final output
                st.markdown(f"**_{mode_label}_**\n\n{response_text}")

        # Add assistant response to chat history.
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# --- Main app layout and sidebar controls -----------------------------------------------

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Ensure session keys exist so sidebar widgets can bind to them.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Detailed"
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""

    # Sidebar: Navigation and controls.
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )

        st.divider()

        st.markdown("## Chat Controls")

        # Response mode selector bound to session state.
        st.selectbox("Response mode:", ["Detailed", "Concise"], key="response_mode", help="Detailed gives longer answers. Concise gives short answers.")

        # System prompt editable by user.
        st.text_area("System prompt (optional):", value=st.session_state.get("system_prompt", ""), key="system_prompt", help="Set the assistant persona or behaviour.")

        st.divider()

        # Upload area for documents to index into ChromaDB.
        st.markdown("### Upload documents to index")
        uploaded_files = st.file_uploader("Upload text files to index (txt, md, pdf). Use 'Index uploaded files' to add them to the vector DB.", type=['txt', 'md', 'pdf'], accept_multiple_files=True)

        if st.button("Index uploaded files"):
            if not uploaded_files:
                st.info("No files selected.")
            else:
                to_index = []
                for file in uploaded_files:
                    try:
                        fname = file.name
                        # Handle PDF using PyPDF2 if available.
                        if fname.lower().endswith(".pdf"):
                            try:
                                import PyPDF2
                                reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                                text_pages = []
                                for p in reader.pages:
                                    try:
                                        text_pages.append(p.extract_text() or "")
                                    except Exception:
                                        text_pages.append("")
                                full_text = "\n".join(text_pages)
                            except Exception:
                                # Fallback: try to decode bytes as text.
                                try:
                                    full_text = file.getvalue().decode("utf-8", errors="ignore")
                                except Exception:
                                    full_text = ""
                        else:
                            # Text/markdown files.
                            try:
                                full_text = file.getvalue().decode("utf-8", errors="ignore")
                            except Exception:
                                full_text = str(file.getvalue())

                        if not full_text.strip():
                            st.warning(f"File `{fname}` yielded no text and will be skipped.")
                            continue

                        # Chunk and prepare docs for indexing.
                        chunks = chunk_text(full_text, chunk_size=500, overlap=50)
                        for i, c in enumerate(chunks):
                            doc = {"id": f"{fname}_chunk_{i}", "text": c, "meta": {"source": fname, "chunk": i}}
                            to_index.append(doc)

                    except Exception as e:
                        st.error(f"Failed to process file {file.name}: {e}")

                if to_index:
                    try:
                        index_documents(to_index, collection_name="docs")
                        st.success(f"Indexed {len(to_index)} chunks from {len(uploaded_files)} files.")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

        st.divider()

        # Clear chat history.
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Route pages.
    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
