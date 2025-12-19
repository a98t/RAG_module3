"""
Enhanced RAG Pipeline V2: Dynamic K with Confidence Threshold

Iteration 2 improvements over V1:
1. Hybrid Search + Reranking (kept from V1)
2. **NEW:** Dynamic K selection based on cross-encoder confidence scores
3. **NEW:** Only return documents above confidence threshold
4. **NEW:** Adaptive: returns 1-5 docs based on relevance, not fixed 5

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


class EnhancedRAGPipelineV2:
    """Enhanced RAG pipeline with hybrid search, reranking, and dynamic K selection."""
    
    def __init__(
        self, 
        initial_k: int = 10,
        min_docs: int = 1,
        max_docs: int = 5,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the enhanced RAG pipeline V2.
        
        Args:
            initial_k: Number of documents to retrieve from each method before reranking
            min_docs: Minimum number of documents to return (even if below threshold)
            max_docs: Maximum number of documents to return
            confidence_threshold: Minimum cross-encoder score to include a document
        """
        self.initial_k = initial_k
        self.min_docs = min_docs
        self.max_docs = max_docs
        self.confidence_threshold = confidence_threshold
        
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
        print(f"⚙️  Configuration: threshold={confidence_threshold}, min={min_docs}, max={max_docs}")
    
    def _load_all_documents(self) -> List[Dict]:
        """Load all documents from Weaviate for BM25 indexing."""
        try:
            collection = self.db_client.get_collection()
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
        """Perform BM25 keyword search."""
        if not self.all_docs:
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.all_docs[idx].copy()
                doc['bm25_score'] = float(scores[idx])
                doc['retrieval_method'] = 'bm25'
                results.append(doc)
        
        return results
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform vector similarity search."""
        try:
            query_vector = embed_text(query)
            results = self.db_client.search_similar_docs(
                query_vector=query_vector,
                limit=top_k
            )
            
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
        """Merge results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        
        # Process vector results
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.get('doc_id', '')
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Process BM25 results
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.get('doc_id', '')
            if doc_id:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
        # Build unified document list
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
        
        merged_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return merged_results
    
    def _rerank_with_dynamic_threshold(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Rerank candidates and apply dynamic K selection based on confidence threshold.
        
        NEW in V2: Only return documents above confidence threshold (between min_docs and max_docs).
        
        Args:
            query: Original query
            candidates: Candidate documents to rerank
            
        Returns:
            Filtered documents that meet confidence threshold (1-5 docs)
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        for doc in candidates:
            doc_text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
            pairs.append([query, doc_text])
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(candidates, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score (descending)
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # **NEW V2 LOGIC:** Dynamic K based on confidence threshold
        selected_docs = []
        
        # Always include at least min_docs (default 1)
        for i in range(min(self.min_docs, len(reranked))):
            selected_docs.append(reranked[i])
        
        # Add additional docs if they meet confidence threshold (up to max_docs)
        for i in range(self.min_docs, min(self.max_docs, len(reranked))):
            if reranked[i]['rerank_score'] >= self.confidence_threshold:
                selected_docs.append(reranked[i])
            else:
                break  # Stop when we hit a doc below threshold
        
        return selected_docs
    
    def answer_question(
        self,
        question: str
    ) -> Tuple[str, List[Dict]]:
        """
        Answer a question using enhanced RAG with dynamic K selection.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (answer_string, list_of_retrieved_documents)
        """
        if not question or not question.strip():
            return "Please ask a question.", []
        
        try:
            # Step 1: Hybrid Retrieval
            print(f"🔍 Searching for relevant documents...")
            print(f"  → Vector search (K={self.initial_k})...")
            vector_results = self._vector_search(question, self.initial_k)
            
            print(f"  → BM25 search (K={self.initial_k})...")
            bm25_results = self._bm25_search(question, self.initial_k)
            
            # Step 2: Merge with RRF
            print(f"  → Merging results with RRF...")
            merged_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
            
            if not merged_results:
                return "I couldn't find any relevant information to answer your question.", []
            
            print(f"  → Found {len(merged_results)} unique candidates")
            
            # Step 3: Rerank with Dynamic Threshold
            print(f"  → Reranking with confidence threshold {self.confidence_threshold}...")
            selected_docs = self._rerank_with_dynamic_threshold(query=question, candidates=merged_results)
            
            print(f"✅ Dynamic selection: {len(selected_docs)} documents (range: {self.min_docs}-{self.max_docs})")
            
            # Step 4: Prepare documents for LLM
            context_docs = []
            for result in selected_docs:
                doc_id_value = result.get("doc_id", "")
                context_docs.append({
                    "id": doc_id_value,
                    "doc_id": doc_id_value,
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "topic": result.get("topic", ""),
                    "similarity": result.get("similarity", result.get("rerank_score", 0.0))
                })
            
            # Step 5: Generate answer
            print(f"💭 Generating answer...")
            answer = ask_llm_with_context(
                question=question,
                context_documents=context_docs
            )
            
            print(f"✅ Answer generated")
            
            return answer, selected_docs
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg, []
    
    def close(self):
        """Close database connection."""
        self.db_client.disconnect()


# Global pipeline instance
_enhanced_pipeline_v2 = None


def get_pipeline(
    initial_k: int = 10,
    min_docs: int = 1,
    max_docs: int = 5,
    confidence_threshold: float = 0.5
) -> EnhancedRAGPipelineV2:
    """Get or create a global enhanced RAG pipeline V2 instance."""
    global _enhanced_pipeline_v2
    if _enhanced_pipeline_v2 is None:
        _enhanced_pipeline_v2 = EnhancedRAGPipelineV2(
            initial_k=initial_k,
            min_docs=min_docs,
            max_docs=max_docs,
            confidence_threshold=confidence_threshold
        )
    return _enhanced_pipeline_v2


def answer_question(question: str) -> Tuple[str, List[Dict]]:
    """Convenience function to answer a question using the global enhanced pipeline V2."""
    pipeline = get_pipeline()
    return pipeline.answer_question(question)


if __name__ == "__main__":
    import os
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    print("=" * 70)
    print("Testing Enhanced RAG Pipeline V2 (Dynamic K)")
    print("=" * 70)
    
    test_questions = [
        "What is the xG metric in football analytics?",
        "When was the Premier League founded?",
    ]
    
    pipeline = EnhancedRAGPipelineV2(
        initial_k=10,
        min_docs=1,
        max_docs=5,
        confidence_threshold=0.5
    )
    
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
                print(f"   Rerank Score: {doc.get('rerank_score', 0):.4f}")
                if doc.get('rerank_score', 0) < 0.5:
                    print(f"   ⚠️  Below threshold (0.5)")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.close()
