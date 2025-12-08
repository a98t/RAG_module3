"""
Premier League Insight Assistant - Streamlit UI

Web interface for the RAG-based Premier League question answering system.
"""

import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="Premier League Insight Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #38003c;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .document-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #38003c;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'history' not in st.session_state:
    st.session_state.history = []


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    if st.session_state.pipeline is None:
        with st.spinner("🔧 Connecting to knowledge base..."):
            try:
                st.session_state.pipeline = RAGPipeline(top_k=5)
                return True
            except Exception as e:
                st.error(f"❌ Error connecting to database: {e}")
                st.info("Make sure Weaviate is running: `docker ps`")
                return False
    return True


def display_header():
    """Display the application header."""
    st.markdown('<div class="main-header">⚽ Premier League Insight Assistant</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subheader">Ask questions about the English Premier League</div>', 
                unsafe_allow_html=True)


def display_sidebar():
    """Display the sidebar with information and settings."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This AI assistant uses **Retrieval-Augmented Generation (RAG)** 
        to answer questions about the Premier League.
        
        **Topics covered:**
        - Origins & Structure
        - Iconic Moments
        - Football Analytics
        - Tactics & Playing Styles
        - Fan Culture
        """)
        
        st.divider()
        
        st.header("⚙️ Settings")
        top_k = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant documents to use for answering"
        )
        
        st.divider()
        
        st.header("📝 Example Questions")
        examples = [
            "What is the xG metric?",
            "Tell me about the Invincibles",
            "How does relegation work?",
            "What was Leicester's miracle season?",
            "Explain the false nine tactic"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.current_question = example
        
        st.divider()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        return top_k


def main():
    """Main application logic."""
    display_header()
    
    # Sidebar
    top_k = display_sidebar()
    
    # Initialize pipeline
    if not initialize_pipeline():
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        # Question input
        question = st.text_area(
            "Your question:",
            value=st.session_state.get('current_question', ''),
            height=100,
            placeholder="E.g., What was the most dramatic title race in Premier League history?",
            key="question_input"
        )
        
        # Clear the stored question after displaying
        if 'current_question' in st.session_state:
            del st.session_state.current_question
        
        # Ask button
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if question.strip():
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Get answer from RAG pipeline
                        answer, docs = st.session_state.pipeline.answer_question(
                            question,
                            top_k=top_k
                        )
                        
                        # Add to history
                        st.session_state.history.insert(0, {
                            'question': question,
                            'answer': answer,
                            'docs': docs
                        })
                        
                        # Clear input
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter a question")
    
    with col2:
        st.subheader("📊 Statistics")
        
        if st.session_state.pipeline:
            try:
                total_docs = st.session_state.pipeline.db_client.get_document_count()
                st.metric("Total Documents", total_docs)
            except:
                st.metric("Total Documents", "N/A")
        
        st.metric("Questions Asked", len(st.session_state.history))
        st.metric("Retrieval Depth", f"Top {top_k}")
    
    # Display history
    if st.session_state.history:
        st.divider()
        st.subheader("📚 Q&A History")
        
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"Q: {item['question'][:80]}...", expanded=(idx == 0)):
                # Question
                st.markdown(f"**Question:** {item['question']}")
                
                # Answer
                st.markdown("**Answer:**")
                st.info(item['answer'])
                
                # Retrieved documents
                if item['docs']:
                    st.markdown(f"**Retrieved Context ({len(item['docs'])} documents):**")
                    
                    for i, doc in enumerate(item['docs'], 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="document-card">
                                <strong>{i}. {doc['title']}</strong><br>
                                <em>Topic: {doc['topic']} | Similarity: {doc['similarity']:.2%}</em><br>
                                <small>{doc['content'][:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.info("👆 Ask a question to get started!")
    
    # Footer
    st.divider()
    st.caption("Powered by RAG | OpenAI Embeddings & GPT-4 | Weaviate Vector Database")


if __name__ == "__main__":
    import os
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY environment variable not set")
        st.info("Please create a .env file with your OpenAI API key")
        st.stop()
    
    main()
