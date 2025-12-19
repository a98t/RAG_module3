import json
import time
from typing import List, Dict, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RAGEvaluator:
    """Automated testing and evaluation for RAG system"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def load_test_dataset(self, filepath: str) -> List[Dict]:
        """Load test questions from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_cases']
    
    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Use LLM-as-judge to rate answer relevance (0-1 scale)
        """
        prompt = f"""Rate how well this answer addresses the question on a scale of 0.0 to 1.0.
        
Question: {question}
Answer: {answer}

Criteria:
- 1.0: Perfect answer, fully addresses the question
- 0.7-0.9: Good answer, addresses most aspects
- 0.4-0.6: Partial answer, missing key information
- 0.1-0.3: Poor answer, barely relevant
- 0.0: Completely irrelevant

Respond with ONLY a number between 0.0 and 1.0."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except:
            return 0.5  # Default if parsing fails
    
    def evaluate_retrieval_precision(self, 
                                     retrieved_docs: List[Dict],
                                     relevant_doc_ids: List[str]) -> float:
        """
        Calculate precision: what fraction of retrieved docs are relevant?
        Precision@K = (# relevant docs retrieved) / K
        """
        if not retrieved_docs:
            return 0.0
        
        retrieved_ids = [doc.get('id', '') for doc in retrieved_docs]
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_doc_ids)
        
        precision = relevant_retrieved / len(retrieved_docs)
        return precision
    
    def evaluate_context_recall(self,
                                retrieved_docs: List[Dict],
                                relevant_doc_ids: List[str]) -> float:
        """
        Calculate recall: what fraction of relevant docs were retrieved?
        Recall = (# relevant docs retrieved) / (# total relevant docs)
        """
        if not relevant_doc_ids:
            return 1.0  # No relevant docs needed
        
        retrieved_ids = [doc.get('id', '') for doc in retrieved_docs]
        relevant_retrieved = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_ids)
        
        recall = relevant_retrieved / len(relevant_doc_ids)
        return recall
    
    def evaluate_faithfulness(self, answer: str, retrieved_docs: List[Dict]) -> float:
        """
        Check if answer is grounded in retrieved context (anti-hallucination)
        """
        context = "\n\n".join([doc.get('content', '') for doc in retrieved_docs])
        
        prompt = f"""Does the answer contain information that is NOT present in the context?

Context:
{context}

Answer:
{answer}

Rate faithfulness on 0.0 to 1.0 scale:
- 1.0: Answer fully grounded in context, no hallucinations
- 0.7-0.9: Mostly grounded, minor inferences
- 0.4-0.6: Some unsupported claims
- 0.1-0.3: Many unsupported claims
- 0.0: Completely fabricated

Respond with ONLY a number between 0.0 and 1.0."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def run_evaluation(self, test_dataset: List[Dict], top_k: int = 5) -> Dict:
        """
        Run full evaluation on test dataset
        Returns aggregated metrics
        """
        results = {
            "total_questions": len(test_dataset),
            "individual_results": [],
            "metrics": {
                "answer_relevance": [],
                "retrieval_precision": [],
                "context_recall": [],
                "faithfulness": [],
                "response_time": []
            }
        }
        
        print(f"🧪 Running evaluation on {len(test_dataset)} test cases...\n")
        
        for i, test_case in enumerate(test_dataset, 1):
            question = test_case['question']
            relevant_doc_ids = test_case.get('relevant_doc_ids', [])
            
            print(f"[{i}/{len(test_dataset)}] Testing: {question[:60]}...")
            
            # Measure response time
            start_time = time.time()
            answer, retrieved_docs = self.rag_pipeline.answer_question(question, top_k=top_k)
            response_time = time.time() - start_time
            
            # Calculate metrics
            relevance_score = self.evaluate_answer_relevance(question, answer)
            precision_score = self.evaluate_retrieval_precision(retrieved_docs, relevant_doc_ids)
            recall_score = self.evaluate_context_recall(retrieved_docs, relevant_doc_ids)
            faithfulness_score = self.evaluate_faithfulness(answer, retrieved_docs)
            
            # Store results
            result = {
                "question_id": test_case['id'],
                "question": question,
                "answer": answer,
                "retrieved_doc_count": len(retrieved_docs),
                "retrieved_doc_ids": [doc.get('id', '') for doc in retrieved_docs],
                "metrics": {
                    "answer_relevance": relevance_score,
                    "retrieval_precision": precision_score,
                    "context_recall": recall_score,
                    "faithfulness": faithfulness_score,
                    "response_time": response_time
                }
            }
            
            results["individual_results"].append(result)
            results["metrics"]["answer_relevance"].append(relevance_score)
            results["metrics"]["retrieval_precision"].append(precision_score)
            results["metrics"]["context_recall"].append(recall_score)
            results["metrics"]["faithfulness"].append(faithfulness_score)
            results["metrics"]["response_time"].append(response_time)
            
            print(f"  ✓ Relevance: {relevance_score:.2f} | Precision: {precision_score:.2f} | "
                  f"Recall: {recall_score:.2f} | Time: {response_time:.2f}s\n")
        
        # Calculate averages
        results["average_metrics"] = {
            metric: sum(scores) / len(scores) if scores else 0.0
            for metric, scores in results["metrics"].items()
        }
        
        return results
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        
        avg = results["average_metrics"]
        
        print(f"\n📈 Average Metrics:")
        print(f"  • Answer Relevance:      {avg['answer_relevance']:.3f} (0-1 scale)")
        print(f"  • Retrieval Precision:   {avg['retrieval_precision']:.3f} (0-1 scale)")
        print(f"  • Context Recall:        {avg['context_recall']:.3f} (0-1 scale)")
        print(f"  • Faithfulness:          {avg['faithfulness']:.3f} (0-1 scale)")
        print(f"  • Avg Response Time:     {avg['response_time']:.2f}s")
        
        print(f"\n📊 Total Questions: {results['total_questions']}")
        print("="*70 + "\n")
    
    def save_results(self, results: Dict, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {filepath}")


def main():
    """Run baseline evaluation"""
    print("🚀 Starting RAG System Baseline Evaluation\n")
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    print("✓ RAG pipeline initialized\n")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(pipeline)
    
    # Load test dataset
    test_cases = evaluator.load_test_dataset('tests/test_dataset.json')
    print(f"✓ Loaded {len(test_cases)} test cases\n")
    
    # Run evaluation
    results = evaluator.run_evaluation(test_cases, top_k=5)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results, 'tests/baseline_results.json')
    
    print("\n✅ Baseline evaluation complete!")
    print("Next steps:")
    print("1. Review baseline_results.json")
    print("2. Identify metrics to improve")
    print("3. Design enhancement strategy")


if __name__ == "__main__":
    main()
