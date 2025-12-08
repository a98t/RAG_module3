"""
Data Ingestion Script

This script loads the Premier League documents from the JSONL file,
generates embeddings for each document, and inserts them into Weaviate.
"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embeddings_client import embed_batch
from db_client import WeaviateClient


def load_documents(file_path: str):
    """
    Load documents from JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    print(f"📖 Loading documents from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():  # Skip empty lines
                    try:
                        doc = json.loads(line)
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Skipping line {line_num} due to JSON error: {e}")
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        sys.exit(1)


def generate_embeddings(documents):
    """
    Generate embeddings for all documents.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        List of embedding vectors in the same order as documents
    """
    print(f"\n🔢 Generating embeddings for {len(documents)} documents...")
    
    try:
        # Extract content from all documents
        contents = [doc['content'] for doc in documents]
        
        # Generate embeddings in batch (more efficient)
        embeddings = embed_batch(contents)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Vector dimension: {len(embeddings[0])}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        sys.exit(1)


def ingest_to_weaviate(documents, embeddings):
    """
    Insert documents and their embeddings into Weaviate.
    
    Args:
        documents: List of document dictionaries
        embeddings: List of embedding vectors
    """
    print(f"\n💾 Ingesting documents into Weaviate...")
    
    db_client = WeaviateClient()
    
    try:
        # Connect to Weaviate
        db_client.connect()
        
        # Create schema (this will delete existing collection if present)
        db_client.create_schema()
        
        # Prepare documents with embeddings for batch insert
        docs_with_vectors = []
        for doc, embedding in zip(documents, embeddings):
            docs_with_vectors.append({
                "doc_id": doc['id'],
                "title": doc['title'],
                "topic": doc['topic'],
                "tags": doc['tags'],
                "content": doc['content'],
                "vector": embedding
            })
        
        # Insert all documents in batch
        count = db_client.insert_documents_batch(docs_with_vectors)
        
        # Verify ingestion
        total_count = db_client.get_document_count()
        print(f"✅ Ingestion complete!")
        print(f"   Documents in database: {total_count}")
        
        return total_count
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        sys.exit(1)
    
    finally:
        db_client.disconnect()


def main():
    """Main ingestion workflow."""
    print("=" * 70)
    print("Premier League Knowledge Base - Data Ingestion")
    print("=" * 70)
    
    # Determine the data file path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_file = project_root / "data" / "premier_league_documents.jsonl"
    
    print(f"\n📁 Project root: {project_root}")
    print(f"📁 Data file: {data_file}")
    
    # Check if data file exists
    if not data_file.exists():
        print(f"❌ Error: Data file not found at {data_file}")
        sys.exit(1)
    
    # Step 1: Load documents
    documents = load_documents(str(data_file))
    
    # Display summary of topics
    topics = {}
    for doc in documents:
        topic = doc['topic']
        topics[topic] = topics.get(topic, 0) + 1
    
    print("\n📊 Dataset summary:")
    for topic, count in sorted(topics.items()):
        print(f"   {topic}: {count} documents")
    
    # Step 2: Generate embeddings
    embeddings = generate_embeddings(documents)
    
    # Step 3: Ingest to Weaviate
    total = ingest_to_weaviate(documents, embeddings)
    
    print("\n" + "=" * 70)
    print(f"✅ SUCCESS! {total} documents are now searchable in Weaviate.")
    print("=" * 70)
    print("\nYou can now run the UI:")
    print("  streamlit run app.py")
    print("  or")
    print("  python app_cli.py")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("\nPlease create a .env file with:")
        print("OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    main()
