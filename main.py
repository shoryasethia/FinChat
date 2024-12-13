import os
import logging
import streamlit as st
import re
import traceback
from typing import List, Dict

from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

# Configure logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Working directory
WORKING_DIR = "output/knowledge_graph"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Predefined overview query
OVERVIEW_QUERY = """
Provide a comprehensive and concise professional summary of this document. 
Focus on the key aspects of the document/report, highlighting 
the most important information about overall status, 
business segments, and critical insights. Structure the summary 
to read like an executive briefing, emphasizing strategic 
implications and key takeaways.
"""

# Function to initialize LightRAG with enhanced error handling
def initialize_rag() -> LightRAG:
    try:
        # Use a more informative loading message
        with st.spinner("🚀 Initializing Advanced Document Analysis Engine..."):
            rag = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=ollama_model_complete,
                llm_model_name="qwen2m",
                llm_model_max_async=4,
                llm_model_max_token_size=32768,
                llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
                embedding_func=EmbeddingFunc(
                    embedding_dim=768,
                    max_token_size=8192,
                    func=lambda texts: ollama_embedding(
                        texts, embed_model="nomic-embed-text", host="http://localhost:11434"
                    ),
                ),
            )
        
        # Success notification with more context
        st.toast("✅ Document Analysis Engine Initialized Successfully!", icon="🎉")
        return rag
    except Exception as e:
        # Detailed error logging and user-friendly error message
        st.error(f"🚨 Initialization Error: {e}")
        st.error("Unable to initialize the Document Analysis Engine. Please check:")
        st.error("1. Ollama is running")
        st.error("2. Model 'qwen2m' and 'nomic-embed-text' are available")
        st.error("3. Local server is accessible")
        
        # Log full traceback for debugging
        logging.error(traceback.format_exc())
        return None

# Enhanced response cleaning function
def clean_response(response: str) -> str:
    # Remove log prefixes, extra whitespaces, and standardize newlines
    cleaned = re.sub(r'(INFO|DEBUG|WARNING):.*\n?', '', response)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Normalize multiple newlines
    return cleaned.strip()

# Enhanced query function with more robust error handling
def perform_query(rag: LightRAG, query: str, mode: str) -> str:
    if not query or query.isspace():
        return "❌ Please enter a valid query."
    
    try:
        with st.spinner(f"🔍 Performing {mode.capitalize()} Search..."):
            query_param = QueryParam(mode=mode)
            response = rag.query(query, param=query_param)
            
            # Enhanced cleaning and validation
            cleaned_response = clean_response(response)
            
            if not cleaned_response:
                return "🤷 No relevant information found. Try a different query."
            
            return cleaned_response
    except Exception as e:
        # Comprehensive error handling
        error_message = f"🚨 Search Error in {mode.capitalize()} Mode:\n"
        error_message += f"- Query: {query}\n"
        error_message += f"- Error Details: {str(e)}"
        
        # Log the full traceback for debugging
        logging.error(traceback.format_exc())
        
        return error_message

def main():
    # Enhanced page configuration
    st.set_page_config(
        page_title="Advanced Document Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for improved aesthetics
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #4A4A4A;
    }
    .chat-container {
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 10px;
    }
    .user-message {
        background-color: #2C3E50;
        color: #ECF0F1;
    }
    .assistant-message {
        background-color: #34495E;
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'rag' not in st.session_state:
        st.session_state.rag = initialize_rag()
    
    # Add a flag to clear input
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False

    # Verify RAG initialization
    if not st.session_state.rag:
        st.error("❌ Document Analysis Engine Initialization Failed")
        return

    # Sidebar with enhanced design
    with st.sidebar:
        st.title("🔍 Document Analyzer Pro")
        
        # More visually appealing buttons
        col1, col2 = st.columns(2)
        with col1:
            overview_btn = st.button("📄 Overview", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

        if overview_btn:
            overview = clean_response(
                st.session_state.rag.query(
                    OVERVIEW_QUERY, 
                    param=QueryParam(mode="global")
                )
            )
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": f"📄 **Document Overview**:\n{overview}",
                "search_mode": "overview"
            })
            st.rerun()

        if clear_btn:
            st.session_state.chat_history = []
            st.rerun()

    # Main content area
    st.header("💬 Document Intelligence")

    # Enhanced chat history display
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-container user-message">
            👤 **You**: {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-container assistant-message">
            🤖 **{message.get('search_mode', 'Search').capitalize()} Result**:
            
            {message['content']}
            </div>
            """, unsafe_allow_html=True)

    # Query input and search modes
    # Use the clear_input flag to conditionally set the default value
    default_query = "" if st.session_state.clear_input else None
    query = st.text_input("Enter your query:", key="query_input", placeholder="Ask something about the document...", value=default_query)

    # Reset the clear_input flag after using it
    st.session_state.clear_input = False

    # Search mode buttons with icons and colors
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        local_search = st.button("🔬 Local", use_container_width=True)
    with col2:
        global_search = st.button("🌐 Global", use_container_width=True)
    with col3:
        naive_search = st.button("📊 Naive", use_container_width=True)
    with col4:
        hybrid_search = st.button("🔮 Hybrid", use_container_width=True)

    # Process search
    if query:
        # Determine search mode
        search_mode_map = {
            local_search: "local",
            global_search: "global",
            naive_search: "naive",
            hybrid_search: "hybrid"
        }
        
        search_mode = next((mode for btn, mode in search_mode_map.items() if btn), None)

        if search_mode:
            # Add user query to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": query
            })

            # Perform the query
            response = perform_query(st.session_state.rag, query, search_mode)
            
            # Add response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "search_mode": search_mode
            })

            # Set flag to clear input on next render
            st.session_state.clear_input = True

            # Rerun to refresh the chat history display
            st.rerun()

if __name__ == "__main__":
    main()