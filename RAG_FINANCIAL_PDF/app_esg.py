"""Streamlit web interface for Financial ESG RAG Chatbot."""
import streamlit as st
import uuid
from loguru import logger
from src.chatbot.esg_chatbot import ESGChatbot
from src.config import config

# Configure logging
logger.add("logs/chatbot.log", rotation="1 day", retention="7 days", level=config.log_level)

# Page configuration
st.set_page_config(
    page_title="Financial ESG Analyst",
    page_icon="📊",
    layout="wide"
)

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot (cached)."""
    return ESGChatbot()

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
        st.session_state.chatbot.create_session(st.session_state.session_id)

# Main app
def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📊 ESG Analyst")
        st.markdown("---")
        
        # System health check
        st.subheader("System Status")
        health = st.session_state.chatbot.health_check()
        
        col1, col2 = st.columns(2)
        with col1:
            status_emoji = "✅" if health["llm_api"] else "❌"
            st.metric("LLM API", status_emoji)
        with col2:
            status_emoji = "✅" if health["qdrant"] else "❌"
            st.metric("Vector DB", status_emoji)
        
        st.markdown("---")
        
        # Session info
        st.subheader("Session Info")
        st.text(f"Session ID:\n{st.session_state.session_id[:18]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chatbot.clear_session(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            **Welcome to Financial ESG Analyst!**
            
            I can help you with:
            - ESG metrics and targets
            - Carbon emissions data
            - Sustainability initiatives
            - Financial ESG reporting
            - Company comparisons
            
            **Available Reports:**
            - Absa Group (2022)
            - Clicks (2022)
            - Distell (2022)
            - Pick n Pay (2023)
            - Sasol (2023)
            
            **Example Questions:**
            - "What are Absa's carbon emissions targets?"
            - "Compare Clicks and Pick n Pay sustainability goals"
            - "What was Sasol's total energy use in 2023?"
            - "Tell me about Distell's ESG initiatives"
            """)
    
    # Main chat interface
    st.title("🤖 Financial ESG Analyst")
    st.markdown("Your intelligent assistant for ESG financial reporting analysis")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about ESG reports..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing ESG reports..."):
                response = st.session_state.chatbot.process_message(
                    message=prompt,
                    session_id=st.session_state.session_id
                )
                st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Force rerun to update UI
        st.rerun()

if __name__ == "__main__":
    main()
