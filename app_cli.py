"""
Premier League Insight Assistant - CLI Version

Command-line interface for the RAG-based Premier League question answering system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline


def print_header():
    """Print the application header."""
    print("\n" + "=" * 70)
    print("⚽ PREMIER LEAGUE INSIGHT ASSISTANT")
    print("=" * 70)
    print("Ask questions about the English Premier League")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'help' for example questions")
    print("=" * 70 + "\n")


def print_help():
    """Print example questions."""
    print("\n📝 Example questions:")
    print("  • What is the xG metric in football analytics?")
    print("  • Tell me about Arsenal's Invincibles season")
    print("  • How does the Premier League relegation system work?")
    print("  • What was Leicester City's miracle championship?")
    print("  • Explain the false nine tactical approach")
    print("  • What is PPDA in pressing analytics?")
    print()


def display_answer(answer: str, docs: list):
    """
    Display the answer and retrieved documents.
    
    Args:
        answer: The generated answer
        docs: List of retrieved documents
    """
    print("\n" + "-" * 70)
    print("📝 ANSWER:")
    print("-" * 70)
    print(answer)
    print()
    
    if docs:
        print("-" * 70)
        print(f"📚 RETRIEVED CONTEXT ({len(docs)} documents):")
        print("-" * 70)
        
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc['title']}")
            print(f"   Topic: {doc['topic']}")
            print(f"   Similarity: {doc['similarity']:.2%}")
            print(f"   Tags: {', '.join(doc['tags'])}")
            print(f"   Preview: {doc['content'][:150]}...")
        
        print()


def main():
    """Main CLI loop."""
    print_header()
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease set your API key:")
        print("  Windows: $env:OPENAI_API_KEY='your-key-here'")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        print("\nOr create a .env file with:")
        print("  OPENAI_API_KEY=your-key-here")
        return
    
    # Initialize RAG pipeline
    print("🔧 Initializing RAG pipeline...")
    
    try:
        pipeline = RAGPipeline(top_k=5)
        print("✅ Ready! Database connected.\n")
    except Exception as e:
        print(f"❌ Error initializing pipeline: {e}")
        print("\nMake sure:")
        print("  1. Weaviate is running: docker ps")
        print("  2. Data is ingested: python scripts/ingest_data.py")
        return
    
    # Main loop
    question_count = 0
    
    try:
        while True:
            # Get user input
            question = input("🔍 Your question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            # Check for help
            if question.lower() in ['help', 'h', '?']:
                print_help()
                continue
            
            # Skip empty questions
            if not question:
                continue
            
            # Process question
            print("\n⏳ Searching and generating answer...")
            
            try:
                answer, docs = pipeline.answer_question(question)
                display_answer(answer, docs)
                question_count += 1
            
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        # Goodbye message
        print("\n" + "=" * 70)
        print(f"Thank you for using Premier League Insight Assistant!")
        print(f"Questions answered: {question_count}")
        print("=" * 70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
