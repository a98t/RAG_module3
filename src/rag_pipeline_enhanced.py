"""
Enhanced RAG Pipeline with Hybrid Search and Cross-Encoder Reranking

This module extends the baseline RAG pipeline with:
1. Hybrid Search: Combines vector search (semantic) + BM25 (keyword matching)
2. Cross-Encoder Reranking: Uses a specialized model to score and filter results
3. Reciprocal Rank Fusion (RRF): Merges results from multiple retrievers

Goal: Improve Retrieval Precision from 22.4% to 29.1%+ (30% improvement)
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Add src to path
sys.path.append(str(Path(__file__).parent))

from embeddings_client import embed_text
from llm_client import ask_llm_with_context
from db_client import WeaviateClient


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with hybrid search and reranking."""
    
    def __init__(self, top_k: int = 5, initial_k: int = 10):
        """
        Initialize the enhanced RAG pipeline.
        
        Args:
            top_k: Final number of documents to return after reranking
            initial_k: Number of documents to retrieve from each method before reranking
        """
        self.top_k = top_k
        self.initial_k = initial_k
        
        # Initialize database client
        self.db_client = WeaviateClient()
        self.db_client.connect()
        
        # Load all documents for BM25 indexing
        print("📚 Loading documents for BM25 indexing...")
        self.all_docs = self._load_all_documents()
        
        # Initialize BM25
        print("🔍 Building BM25 index...")
        self.bm25_corpus = [doc['content'].lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        print(f"✅ BM25 index built with {len(self.all_docs)} documents")
        
        # Initialize cross-encoder for reranking
        print("🧠 Loading cross-encoder model...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Cross-encoder loaded")
    
    def _load_all_documents(self) -> List[Dict]:
        """Load all documents from Weaviate for BM25 indexing."""
        try:
            collection = self.db_client.get_collection()
            
            # Query all documents
            response = collection.query.fetch_objects(limit=1000)
            
            docs = []
            for obj in response.objects:
                docs.append({
                    'doc_id': obj.properties.get('doc_id', ''),
                    'title': obj.properties.get('title', ''),
                    'content': obj.properties.get('content', ''),
                    'topic': obj.properties.get('topic', ''),
                    'tags': obj.properties.get('tags', [])
                })
            
            return docs
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load documents for BM25: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of documents with BM25 scores
        """
        if not self.all_docs:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top K indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Build results with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include docs with positive scores
                doc = self.all_docs[idx].copy()
                doc['bm25_score'] = float(scores[idx])
                doc['retrieval_method'] = 'bm25'
                results.append(doc)
        
        return results
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Convert query to embedding
            query_vector = embed_text(query)
            
            # Search Weaviate
            results = self.db_client.search_similar_docs(
                query_vector=query_vector,
                limit=top_k
            )
            
            # Add retrieval method tag
            for doc in results:
                doc['retrieval_method'] = 'vector'
            
            return results
            
        except Exception as e:
            print(f"⚠️  Warning: Vector search failed: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).
        
        Formula: RRF_score = Σ(1 / (k + rank_i))
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: Constant for RRF (default 60)
            
        Returns:
            Merged and deduplicated results sorted by RRF score
        """
        # Build RRF scores
        rrf_scores = {}
        
        # Process vector results (rank by position)
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.get('doc_id', '')
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Process BM25 results (rank by position)
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.get('doc_id', '')
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Build unified document list (deduplicate by doc_id)
        doc_map = {}
        for doc in vector_results + bm25_results:
            doc_id = doc.get('doc_id', '')
            if doc_id and doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Add RRF scores and sort
        merged_results = []
        for doc_id, doc in doc_map.items():
            doc['rrf_score'] = rrf_scores.get(doc_id, 0)
            merged_results.append(doc)
        
        # Sort by RRF score (descending)
        merged_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return merged_results
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Rerank candidates using cross-encoder model.
        
        Args:
            query: Original query
            candidates: Candidate documents to rerank
            top_k: Number of top results to keep
            
        Returns:
            Top K documents sorted by cross-encoder relevance score
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        for doc in candidates:
            # Use title + content for better context
            doc_text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
            pairs.append([query, doc_text])
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(candidates, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score (descending)
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top K
        return reranked[:top_k]
    
    def answer_question(
        self,
        question: str,
        top_k: int = None
    ) -> Tuple[str, List[Dict]]:
        """
        Answer a question using enhanced RAG with hybrid search and reranking.
        
        Args:
            question: The user's question
            top_k: Number of final docs to retrieve (overrides default)
            
        Returns:
            Tuple of (answer_string, list_of_retrieved_documents)
        """
        if not question or not question.strip():
            return "Please ask a question.", []
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Step 1: Hybrid Retrieval (Vector + BM25)
            print(f"🔍 Searching for relevant documents...")
            print(f"  → Vector search (K={self.initial_k})...")
            vector_results = self._vector_search(question, self.initial_k)
            
            print(f"  → BM25 search (K={self.initial_k})...")
            bm25_results = self._bm25_search(question, self.initial_k)
            
            # Step 2: Merge with Reciprocal Rank Fusion
            print(f"  → Merging results with RRF...")
            merged_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
            
            if not merged_results:
                return "I couldn't find any relevant information to answer your question.", []
            
            print(f"  → Found {len(merged_results)} unique candidates")
            
            # Step 3: Rerank with Cross-Encoder
            print(f"  → Reranking with cross-encoder...")
            reranked_results = self._rerank_with_cross_encoder(
                query=question,
                candidates=merged_results,
                top_k=k
            )
            
            print(f"✅ Final selection: {len(reranked_results)} documents")
            
            # Step 4: Prepare documents for LLM
            context_docs = []
            for result in reranked_results:
                doc_id_value = result.get("doc_id", "")
                context_docs.append({
                    "id": doc_id_value,  # For evaluation script
                    "doc_id": doc_id_value,  # Keep original key
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "topic": result.get("topic", ""),
                    "similarity": result.get("similarity", result.get("rerank_score", 0.0))
                })
            
            # Step 5: Generate answer using LLM
            print(f"💭 Generating answer...")
            answer = ask_llm_with_context(
                question=question,
                context_documents=context_docs
            )
            
            print(f"✅ Answer generated")
            
            # Return answer and retrieved docs (with all metadata)
            return answer, reranked_results
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg, []
    
    def close(self):
        """Close database connection."""
        self.db_client.disconnect()


# Global pipeline instance (for simple imports)
_enhanced_pipeline = None


def get_pipeline(top_k: int = 5, initial_k: int = 10) -> EnhancedRAGPipeline:
    """
    Get or create a global enhanced RAG pipeline instance.
    
    Args:
        top_k: Final number of documents to return
        initial_k: Initial number to retrieve before reranking
        
    Returns:
        EnhancedRAGPipeline instance
    """
    global _enhanced_pipeline
    if _enhanced_pipeline is None:
        _enhanced_pipeline = EnhancedRAGPipeline(top_k=top_k, initial_k=initial_k)
    return _enhanced_pipeline


def answer_question(question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    """
    Convenience function to answer a question using the global enhanced pipeline.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        
    Returns:
        Tuple of (answer, retrieved_documents)
    """
    pipeline = get_pipeline(top_k)
    return pipeline.answer_question(question, top_k)


if __name__ == "__main__":
    # Test the enhanced RAG pipeline
    import os
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    print("=" * 70)
    print("Testing Enhanced RAG Pipeline")
    print("=" * 70)
    
    # Test questions
    test_questions = [
        "What is the xG metric in football analytics?",
        "Tell me about Leicester City's championship season.",
    ]
    
    pipeline = EnhancedRAGPipeline(top_k=5, initial_k=10)
    
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
                print(f"\n{j}. {doc.get('title', 'N/A')}")
                print(f"   Doc ID: {doc.get('doc_id', 'N/A')}")
                print(f"   Method: {doc.get('retrieval_method', 'N/A')}")
                print(f"   RRF Score: {doc.get('rrf_score', 0):.4f}")
                print(f"   Rerank Score: {doc.get('rerank_score', 0):.4f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.close()
