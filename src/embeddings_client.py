"""
Embeddings Client Module

This module provides functions to generate vector embeddings from text
using the OpenAI Embeddings API.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def embed_text(text: str) -> List[float]:
    """
    Convert a single text string to a vector embedding.
    
    Args:
        text: The text to embed
        
    Returns:
        A list of floats representing the embedding vector
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Convert multiple text strings to vector embeddings in a single API call.
    More efficient than calling embed_text() multiple times.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        A list of embedding vectors
    """
    if not texts:
        raise ValueError("Text list cannot be empty")
    
    # Filter out empty strings
    valid_texts = [t for t in texts if t and t.strip()]
    if not valid_texts:
        raise ValueError("All texts are empty")
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=valid_texts,
            dimensions=EMBEDDING_DIMENSIONS
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        raise


def get_embedding_dimension() -> int:
    """
    Returns the dimension of the embedding vectors.
    
    Returns:
        Integer representing the vector dimension
    """
    return EMBEDDING_DIMENSIONS


if __name__ == "__main__":
    # Test the embeddings client
    print("Testing Embeddings Client...")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Dimensions: {EMBEDDING_DIMENSIONS}\n")
    
    # Test single embedding
    test_text = "The Premier League is the top tier of English football."
    print(f"Test text: {test_text}")
    
    try:
        embedding = embed_text(test_text)
        print(f"✅ Embedding generated successfully")
        print(f"   Vector length: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}\n")
        
        # Test batch embedding
        test_batch = [
            "Manchester City won the 2017-18 season with 100 points.",
            "Arsenal went unbeaten in the 2003-04 season.",
            "Leicester City won the league in 2015-16."
        ]
        print(f"Testing batch embedding with {len(test_batch)} texts...")
        batch_embeddings = embed_batch(test_batch)
        print(f"✅ Batch embeddings generated successfully")
        print(f"   Number of vectors: {len(batch_embeddings)}")
        print(f"   Each vector length: {len(batch_embeddings[0])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
