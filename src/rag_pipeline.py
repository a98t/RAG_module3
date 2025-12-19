"""
RAG Pipeline Module

This module implements the core RAG (Retrieval-Augmented Generation) workflow:
1. Take a user question
2. Convert it to an embedding
3. Search for similar documents in Weaviate
4. Pass the question + retrieved docs to the LLM
5. Return the generated answer
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent))

from embeddings_client import embed_text
from llm_client import ask_llm_with_context
from db_client import WeaviateClient


class RAGPipeline:
    """Encapsulates the RAG workflow."""
    
    def __init__(self, top_k: int = 5):
        """
        Initialize the RAG pipeline.
        
        Args:
            top_k: Number of documents to retrieve for context
        """
        self.top_k = top_k
        self.db_client = WeaviateClient()
        self.db_client.connect()
    
    def answer_question(
        self,
        question: str,
        top_k: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Answer a question using RAG.
        
        Args:
            question: The user's question
            top_k: Number of docs to retrieve (overrides default)
            
        Returns:
            Tuple of (answer_string, list_of_retrieved_documents)
        """
        if not question or not question.strip():
            return "Please ask a question.", []
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Step 1: Convert question to embedding
            print(f"🔍 Searching for relevant documents...")
            query_vector = embed_text(question)
            
            # Step 2: Search for similar documents
            results = self.db_client.search_similar_docs(
                query_vector=query_vector,
                limit=k
            )
            
            if not results:
                return "I couldn't find any relevant information to answer your question.", []
            
            print(f"✅ Found {len(results)} relevant documents")
            
            # Step 3: Prepare documents for LLM and evaluation
            context_docs = []
            for result in results:
                # Include both 'id' and 'doc_id' for compatibility
                doc_id_value = result.get("doc_id", "")
                context_docs.append({
                    "id": doc_id_value,  # For evaluation script
                    "doc_id": doc_id_value,  # Keep original key
                    "title": result["title"],
                    "content": result["content"],
                    "topic": result.get("topic", ""),
                    "similarity": result.get("similarity", 0.0)
                })
            
            # Step 4: Generate answer using LLM with context
            print(f"💭 Generating answer...")
            answer = ask_llm_with_context(
                question=question,
                context_documents=context_docs
            )
            
            print(f"✅ Answer generated")
            
            # Return answer and retrieved docs (with metadata)
            return answer, results
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, []
    
    def close(self):
        """Close database connection."""
        self.db_client.disconnect()


# Global pipeline instance (for simple imports)
_pipeline = None


def get_pipeline(top_k: int = 5) -> RAGPipeline:
    """
    Get or create a global RAG pipeline instance.
    
    Args:
        top_k: Number of documents to retrieve
        
    Returns:
        RAGPipeline instance
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(top_k=top_k)
    return _pipeline


def answer_question(question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    """
    Convenience function to answer a question using the global pipeline.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (answer, retrieved_documents)
    """
    pipeline = get_pipeline(top_k)
    return pipeline.answer_question(question, top_k)


if __name__ == "__main__":
    # Test the RAG pipeline
    import os
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    print("=" * 70)
    print("Testing RAG Pipeline")
    print("=" * 70)
    
    # Test questions
    test_questions = [
        "What is the xG metric in football analytics?",
        "Tell me about Leicester City's championship season.",
        "How does the Premier League points system work?"
    ]
    
    pipeline = RAGPipeline(top_k=3)
    
    try:
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'=' * 70}")
            print(f"Question {i}: {question}")
            print('=' * 70)
            
            answer, docs = pipeline.answer_question(question)
            
            print(f"\n📝 Answer:")
            print(answer)
            
            print(f"\n📚 Retrieved documents ({len(docs)}):")
            for j, doc in enumerate(docs, 1):
                print(f"\n{j}. {doc['title']}")
                print(f"   Topic: {doc['topic']}")
                print(f"   Similarity: {doc['similarity']:.4f}")
                print(f"   Content preview: {doc['content'][:100]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        pipeline.close()
