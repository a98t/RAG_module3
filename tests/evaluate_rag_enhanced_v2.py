"""
Enhanced RAG System V2 Evaluation Script

Evaluates the V2 system (dynamic K with confidence threshold) using the same
test dataset for fair comparison with baseline and V1.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import importlib

# Force reload of modules
if 'src.rag_pipeline_enhanced_v2' in sys.modules:
    importlib.reload(sys.modules['src.rag_pipeline_enhanced_v2'])
if 'src.embeddings_client' in sys.modules:
    importlib.reload(sys.modules['src.embeddings_client'])
if 'src.llm_client' in sys.modules:
    importlib.reload(sys.modules['src.llm_client'])
if 'src.db_client' in sys.modules:
    importlib.reload(sys.modules['src.db_client'])

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline_enhanced_v2 import EnhancedRAGPipelineV2
from openai import OpenAI

client = OpenAI()


class EnhancedRAGEvaluatorV2:
    """Evaluates the enhanced RAG system V2."""
    
    def __init__(self, test_dataset_path: str, confidence_threshold: float = 0.5):
        self.test_dataset_path = test_dataset_path
        self.test_cases = self._load_test_dataset()
        self.rag_pipeline = EnhancedRAGPipelineV2(
            initial_k=10,
            min_docs=1,
            max_docs=5,
            confidence_threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
    
    def _load_test_dataset(self) -> List[Dict]:
        with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_cases']
    
    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
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
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"  ⚠️  Warning: Could not evaluate answer relevance: {e}")
            return 0.5
    
    def evaluate_retrieval_precision(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
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
        if not relevant_doc_ids:
            return 1.0
        
        relevant_set = set(relevant_doc_ids)
        retrieved_set = set(retrieved_doc_ids)
        
        true_positives = len(relevant_set & retrieved_set)
        recall = true_positives / len(relevant_set)
        
        return recall
    
    def evaluate_faithfulness(self, answer: str, context_docs: List[Dict]) -> float:
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
        results = []
        
        print(f"\n🧪 Running V2 evaluation on {len(self.test_cases)} test cases...\n")
        
        for i, test_case in enumerate(self.test_cases, 1):
            question = test_case['question']
            expected_answer = test_case['expected_answer']
            relevant_doc_ids = test_case['relevant_doc_ids']
            
            print(f"[{i}/{len(self.test_cases)}] Testing: {question[:50]}...")
            
            start_time = time.time()
            answer, retrieved_docs = self.rag_pipeline.answer_question(question)
            response_time = time.time() - start_time
            
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
            
            result = {
                'test_id': test_case['id'],
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': answer,
                'relevant_doc_ids': relevant_doc_ids,
                'retrieved_doc_ids': retrieved_doc_ids,
                'num_retrieved': len(retrieved_doc_ids),  # NEW: track dynamic K
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
            
            print(f"  ✓ K={len(retrieved_doc_ids)} | Rel: {relevance:.2f} | Prec: {precision:.2f} | Recall: {recall:.2f} | Time: {response_time:.2f}s")
            print(f"  📄 Retrieved IDs: {retrieved_doc_ids[:3]}{'...' if len(retrieved_doc_ids) > 3 else ''}")
        
        # Calculate averages
        avg_metrics = {
            'answer_relevance': sum(r['metrics']['answer_relevance'] for r in results) / len(results),
            'retrieval_precision': sum(r['metrics']['retrieval_precision'] for r in results) / len(results),
            'context_recall': sum(r['metrics']['context_recall'] for r in results) / len(results),
            'faithfulness': sum(r['metrics']['faithfulness'] for r in results) / len(results),
            'response_time': sum(r['metrics']['response_time'] for r in results) / len(results),
            'avg_docs_retrieved': sum(r['num_retrieved'] for r in results) / len(results)
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_type': 'enhanced_v2',
            'system_description': 'Hybrid Search + Reranking + Dynamic K (confidence threshold)',
            'confidence_threshold': self.confidence_threshold,
            'test_count': len(results),
            'results': results,
            'average_metrics': avg_metrics
        }
    
    def print_summary(self, evaluation_results: Dict):
        metrics = evaluation_results['average_metrics']
        baseline_precision = 0.224  # From baseline
        
        print("\n" + "=" * 70)
        print("📊 ENHANCED SYSTEM V2 EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\n📈 Average Metrics:")
        print(f"  • Answer Relevance:      {metrics['answer_relevance']:.3f} (0-1 scale)")
        print(f"  • Retrieval Precision:   {metrics['retrieval_precision']:.3f} (0-1 scale)")
        print(f"  • Context Recall:        {metrics['context_recall']:.3f} (0-1 scale)")
        print(f"  • Faithfulness:          {metrics['faithfulness']:.3f} (0-1 scale)")
        print(f"  • Avg Response Time:     {metrics['response_time']:.2f}s")
        print(f"  • Avg Docs Retrieved:    {metrics['avg_docs_retrieved']:.1f} (dynamic: 1-5)")
        
        print(f"\n📊 Comparison with Baseline:")
        improvement = ((metrics['retrieval_precision'] - baseline_precision) / baseline_precision) * 100
        print(f"  • Baseline Precision:    {baseline_precision:.3f} (22.4%)")
        print(f"  • Current Precision:     {metrics['retrieval_precision']:.3f} ({metrics['retrieval_precision']*100:.1f}%)")
        print(f"  • Improvement:           {improvement:+.1f}%")
        
        if metrics['retrieval_precision'] >= 0.291:  # 30% improvement target
            print(f"\n🟢 SUCCESS! Achieved 30%+ improvement target ({improvement:.1f}% > 30%)")
        elif improvement > 0:
            print(f"\n🟡 PARTIAL SUCCESS: Improved but below 30% target ({improvement:.1f}% < 30%)")
        else:
            print(f"\n🔴 FAILED: No improvement over baseline ({improvement:.1f}%)")
        
        print("=" * 70)
    
    def close(self):
        self.rag_pipeline.close()


def main():
    print("🚀 Starting Enhanced RAG System V2 Evaluation\n")
    
    test_dataset_path = Path(__file__).parent / "test_dataset.json"
    results_path = Path(__file__).parent / "enhanced_results_v2.json"
    
    # Try different thresholds to find the best one
    print("Initializing enhanced RAG pipeline V2...")
    print("Testing with confidence threshold = 0.5\n")
    
    evaluator = EnhancedRAGEvaluatorV2(str(test_dataset_path), confidence_threshold=0.5)
    print(f"✓ Enhanced V2 pipeline initialized\n")
    print(f"✓ Loaded {len(evaluator.test_cases)} test cases\n")
    
    try:
        results = evaluator.run_evaluation()
        evaluator.print_summary(results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {results_path}")
        print("\n✅ Enhanced V2 system evaluation complete!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
