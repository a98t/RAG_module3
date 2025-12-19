"""
Enhanced RAG System Evaluation Script

Evaluates the enhanced RAG system (with hybrid search + reranking) using the same
test dataset and metrics as the baseline evaluation for fair comparison.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import importlib

# Force reload of modules to get latest code
if 'src.rag_pipeline_enhanced' in sys.modules:
    importlib.reload(sys.modules['src.rag_pipeline_enhanced'])
if 'src.embeddings_client' in sys.modules:
    importlib.reload(sys.modules['src.embeddings_client'])
if 'src.llm_client' in sys.modules:
    importlib.reload(sys.modules['src.llm_client'])
if 'src.db_client' in sys.modules:
    importlib.reload(sys.modules['src.db_client'])

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline_enhanced import EnhancedRAGPipeline
from openai import OpenAI

# Initialize OpenAI client for evaluation
client = OpenAI()


class EnhancedRAGEvaluator:
    """Evaluates the enhanced RAG system."""
    
    def __init__(self, test_dataset_path: str):
        """
        Initialize evaluator.
        
        Args:
            test_dataset_path: Path to test dataset JSON file
        """
        self.test_dataset_path = test_dataset_path
        self.test_cases = self._load_test_dataset()
        self.rag_pipeline = EnhancedRAGPipeline(top_k=5, initial_k=10)
    
    def _load_test_dataset(self) -> List[Dict]:
        """Load test dataset from JSON file."""
        with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_cases']
    
    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question using LLM-as-judge.
        
        Args:
            question: Original question
            answer: Generated answer
            
        Returns:
            Relevance score between 0 and 1
        """
        prompt = f"""Rate how well this answer addresses the question on a scale of 0.0 to 1.0.

Question: {question}

Answer: {answer}

Provide only a numeric score (e.g., 0.85) without explanation. Consider:
- Does the answer directly address the question?
- Is it complete and informative?
- Is it on-topic?

Score:"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"  ⚠️  Warning: Could not evaluate answer relevance: {e}")
            return 0.5
    
    def evaluate_retrieval_precision(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate retrieval precision: what percentage of retrieved docs are relevant?
        
        Args:
            retrieved_doc_ids: IDs of documents retrieved by the system
            relevant_doc_ids: IDs of actually relevant documents (ground truth)
            
        Returns:
            Precision score between 0 and 1
        """
        if not retrieved_doc_ids:
            return 0.0
        
        relevant_set = set(relevant_doc_ids)
        retrieved_set = set(retrieved_doc_ids)
        
        true_positives = len(relevant_set & retrieved_set)
        precision = true_positives / len(retrieved_set)
        
        return precision
    
    def evaluate_context_recall(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate context recall: what percentage of relevant docs were retrieved?
        
        Args:
            retrieved_doc_ids: IDs of documents retrieved by the system
            relevant_doc_ids: IDs of actually relevant documents (ground truth)
            
        Returns:
            Recall score between 0 and 1
        """
        if not relevant_doc_ids:
            return 1.0  # No relevant docs exist, so perfect recall
        
        relevant_set = set(relevant_doc_ids)
        retrieved_set = set(retrieved_doc_ids)
        
        true_positives = len(relevant_set & retrieved_set)
        recall = true_positives / len(relevant_set)
        
        return recall
    
    def evaluate_faithfulness(self, answer: str, context_docs: List[Dict]) -> float:
        """
        Evaluate if the answer is faithful to the provided context (no hallucination).
        
        Args:
            answer: Generated answer
            context_docs: Documents used as context
            
        Returns:
            Faithfulness score between 0 and 1
        """
        # Combine context
        context_text = "\n\n".join([
            f"Document: {doc.get('title', 'N/A')}\n{doc.get('content', '')}"
            for doc in context_docs
        ])
        
        prompt = f"""Rate how faithful this answer is to the provided context on a scale of 0.0 to 1.0.

Context:
{context_text[:2000]}  

Answer:
{answer}

Provide only a numeric score (e.g., 0.90) without explanation. Consider:
- Are all claims in the answer supported by the context?
- Is there any hallucinated information?
- Does it stay within the bounds of the provided context?

Score:"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"  ⚠️  Warning: Could not evaluate faithfulness: {e}")
            return 0.5
    
    def run_evaluation(self) -> Dict:
        """
        Run complete evaluation on all test cases.
        
        Returns:
            Dictionary containing evaluation results
        """
        results = []
        
        print(f"\n🧪 Running evaluation on {len(self.test_cases)} test cases...\n")
        
        for i, test_case in enumerate(self.test_cases, 1):
            question = test_case['question']
            expected_answer = test_case['expected_answer']
            relevant_doc_ids = test_case['relevant_doc_ids']
            
            print(f"[{i}/{len(self.test_cases)}] Testing: {question[:50]}...")
            
            # Time the RAG pipeline
            start_time = time.time()
            answer, retrieved_docs = self.rag_pipeline.answer_question(question)
            response_time = time.time() - start_time
            
            # Extract retrieved doc IDs (try both 'id' and 'doc_id' keys)
            retrieved_doc_ids = []
            for doc in retrieved_docs:
                doc_id = doc.get('id') or doc.get('doc_id', '')
                if doc_id:
                    retrieved_doc_ids.append(doc_id)
            
            # Calculate metrics
            relevance = self.evaluate_answer_relevance(question, answer)
            precision = self.evaluate_retrieval_precision(retrieved_doc_ids, relevant_doc_ids)
            recall = self.evaluate_context_recall(retrieved_doc_ids, relevant_doc_ids)
            faithfulness = self.evaluate_faithfulness(answer, retrieved_docs)
            
            # Store results
            result = {
                'test_id': test_case['id'],
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': answer,
                'relevant_doc_ids': relevant_doc_ids,
                'retrieved_doc_ids': retrieved_doc_ids,
                'metrics': {
                    'answer_relevance': round(relevance, 3),
                    'retrieval_precision': round(precision, 3),
                    'context_recall': round(recall, 3),
                    'faithfulness': round(faithfulness, 3),
                    'response_time': round(response_time, 2)
                },
                'category': test_case['category'],
                'difficulty': test_case['difficulty']
            }
            results.append(result)
            
            # Debug output
            print(f"  ✓ Relevance: {relevance:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | Time: {response_time:.2f}s")
            print(f"  📄 Retrieved IDs: {retrieved_doc_ids[:3]}{'...' if len(retrieved_doc_ids) > 3 else ''}")
        
        # Calculate averages
        avg_metrics = {
            'answer_relevance': sum(r['metrics']['answer_relevance'] for r in results) / len(results),
            'retrieval_precision': sum(r['metrics']['retrieval_precision'] for r in results) / len(results),
            'context_recall': sum(r['metrics']['context_recall'] for r in results) / len(results),
            'faithfulness': sum(r['metrics']['faithfulness'] for r in results) / len(results),
            'response_time': sum(r['metrics']['response_time'] for r in results) / len(results)
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_type': 'enhanced',
            'system_description': 'Hybrid Search (Vector + BM25) + Cross-Encoder Reranking',
            'test_count': len(results),
            'results': results,
            'average_metrics': avg_metrics
        }
    
    def print_summary(self, evaluation_results: Dict):
        """Print evaluation summary."""
        metrics = evaluation_results['average_metrics']
        
        print("\n" + "=" * 70)
        print("📊 ENHANCED SYSTEM EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\n📈 Average Metrics:")
        print(f"  • Answer Relevance:      {metrics['answer_relevance']:.3f} (0-1 scale)")
        print(f"  • Retrieval Precision:   {metrics['retrieval_precision']:.3f} (0-1 scale)")
        print(f"  • Context Recall:        {metrics['context_recall']:.3f} (0-1 scale)")
        print(f"  • Faithfulness:          {metrics['faithfulness']:.3f} (0-1 scale)")
        print(f"  • Avg Response Time:     {metrics['response_time']:.2f}s")
        
        print(f"\n📊 Total Questions: {evaluation_results['test_count']}")
        
        # Performance warnings
        if metrics['retrieval_precision'] < 0.3:
            print(f"\n🔴 LOW Retrieval Precision ({metrics['retrieval_precision']:.2f}) - Enhancement may not have worked")
        elif metrics['retrieval_precision'] > 0.29:
            print(f"\n🟢 IMPROVED Retrieval Precision ({metrics['retrieval_precision']:.2f}) - Target achieved!")
        
        if metrics['context_recall'] < 0.95:
            print(f"⚠️  Context Recall dropped to {metrics['context_recall']:.2f} - May need to adjust initial_k")
        
        print("=" * 70)
    
    def close(self):
        """Clean up resources."""
        self.rag_pipeline.close()


def main():
    """Main evaluation script."""
    print("🚀 Starting Enhanced RAG System Evaluation\n")
    
    # Paths
    test_dataset_path = Path(__file__).parent / "test_dataset.json"
    results_path = Path(__file__).parent / "enhanced_results.json"
    
    # Initialize evaluator
    print("Initializing enhanced RAG pipeline...")
    evaluator = EnhancedRAGEvaluator(str(test_dataset_path))
    print(f"✓ Enhanced pipeline initialized\n")
    print(f"✓ Loaded {len(evaluator.test_cases)} test cases\n")
    
    try:
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Print summary
        evaluator.print_summary(results)
        
        # Save results
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_path}")
        print("\n✅ Enhanced system evaluation complete!")
        print("\nNext steps:")
        print("1. Compare enhanced_results.json with baseline_results.json")
        print("2. Calculate improvement percentage")
        print("3. Update RAG_ENHANCEMENT_REPORT.md with findings")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
