"""
Database Client Module

This module provides functions to interact with Weaviate vector database
for storing and retrieving Premier League documents.
"""

import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5
from typing import List, Dict, Optional
import json


# Configuration
WEAVIATE_URL = "http://localhost:8080"
COLLECTION_NAME = "PremierLeagueDoc"


class WeaviateClient:
    """Client for interacting with Weaviate vector database."""
    
    def __init__(self, url: str = WEAVIATE_URL):
        """
        Initialize connection to Weaviate.
        
        Args:
            url: Weaviate instance URL
        """
        self.url = url
        self.client = None
        self.collection = None
        
    def connect(self):
        """Establish connection to Weaviate."""
        try:
            self.client = weaviate.connect_to_local(
                host="localhost",
                port=8080
            )
            if self.client.is_ready():
                print(f"✅ Connected to Weaviate at {self.url}")
                return True
            else:
                print(f"❌ Weaviate is not ready")
                return False
        except Exception as e:
            print(f"❌ Error connecting to Weaviate: {e}")
            raise
    
    def disconnect(self):
        """Close connection to Weaviate."""
        if self.client:
            self.client.close()
            print("✅ Disconnected from Weaviate")
    
    def create_schema(self):
        """
        Create the PremierLeagueDoc collection schema if it doesn't exist.
        If it exists, delete and recreate it.
        """
        try:
            # Delete existing collection if it exists
            if self.client.collections.exists(COLLECTION_NAME):
                self.client.collections.delete(COLLECTION_NAME)
                print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'")
            
            # Create new collection
            self.collection = self.client.collections.create(
                name=COLLECTION_NAME,
                properties=[
                    wvc.config.Property(name="doc_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="topic", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="tags", data_type=wvc.config.DataType.TEXT_ARRAY),
                    wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                ],
                # Configure vector with explicit name for better compatibility
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                )
            )
            print(f"✅ Created collection '{COLLECTION_NAME}'")
            
        except Exception as e:
            print(f"❌ Error creating schema: {e}")
            raise
    
    def get_collection(self):
        """Get reference to the collection."""
        if not self.collection:
            self.collection = self.client.collections.get(COLLECTION_NAME)
        return self.collection
    
    def insert_document(
        self,
        doc_id: str,
        title: str,
        topic: str,
        tags: List[str],
        content: str,
        vector: List[float]
    ) -> str:
        """
        Insert a single document into Weaviate.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            topic: Topic category
            tags: List of tags
            content: Document content
            vector: Embedding vector
            
        Returns:
            UUID of the inserted object
        """
        try:
            collection = self.get_collection()
            
            uuid = collection.data.insert(
                properties={
                    "doc_id": doc_id,
                    "title": title,
                    "topic": topic,
                    "tags": tags,
                    "content": content
                },
                vector=vector,
                uuid=generate_uuid5(doc_id)
            )
            
            return str(uuid)
            
        except Exception as e:
            print(f"❌ Error inserting document {doc_id}: {e}")
            raise
    
    def insert_documents_batch(self, documents: List[Dict]) -> int:
        """
        Insert multiple documents in batch mode for better performance.
        
        Args:
            documents: List of dicts with keys: doc_id, title, topic, tags, content, vector
            
        Returns:
            Number of successfully inserted documents
        """
        try:
            collection = self.get_collection()
            count = 0
            
            with collection.batch.dynamic() as batch:
                for doc in documents:
                    batch.add_object(
                        properties={
                            "doc_id": doc["doc_id"],
                            "title": doc["title"],
                            "topic": doc["topic"],
                            "tags": doc["tags"],
                            "content": doc["content"]
                        },
                        vector=doc["vector"],
                        uuid=generate_uuid5(doc["doc_id"])
                    )
                    count += 1
            
            print(f"✅ Inserted {count} documents in batch")
            return count
            
        except Exception as e:
            print(f"❌ Error in batch insert: {e}")
            raise
    
    def search_similar_docs(
        self,
        query_vector: List[float],
        limit: int = 5,
        return_properties: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for documents similar to the query vector.
        
        Args:
            query_vector: The embedding vector to search with
            limit: Maximum number of results to return
            return_properties: List of properties to return (None = all)
            
        Returns:
            List of dicts with document data and similarity scores
        """
        try:
            collection = self.get_collection()
            
            # For Weaviate v4 with default vector configuration
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            
            results = []
            for obj in response.objects:
                result = {
                    "doc_id": obj.properties.get("doc_id"),
                    "title": obj.properties.get("title"),
                    "topic": obj.properties.get("topic"),
                    "tags": obj.properties.get("tags", []),
                    "content": obj.properties.get("content"),
                    "distance": obj.metadata.distance,
                    "similarity": 1 - obj.metadata.distance  # Convert distance to similarity
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            Number of documents
        """
        try:
            collection = self.get_collection()
            return len(collection)
        except Exception as e:
            print(f"❌ Error getting document count: {e}")
            return 0
    
    def delete_all_documents(self):
        """Delete all documents from the collection."""
        try:
            if self.client.collections.exists(COLLECTION_NAME):
                self.client.collections.delete(COLLECTION_NAME)
                print(f"✅ Deleted all documents from '{COLLECTION_NAME}'")
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            raise


if __name__ == "__main__":
    # Test the database client
    print("Testing Database Client...")
    
    db_client = WeaviateClient()
    
    try:
        # Connect
        db_client.connect()
        
        # Create schema
        db_client.create_schema()
        
        # Test insert
        print("\nTesting document insertion...")
        test_vector = [0.1] * 1536  # Dummy vector for testing
        
        uuid = db_client.insert_document(
            doc_id="test_001",
            title="Test Document",
            topic="Testing",
            tags=["test", "example"],
            content="This is a test document for the database client.",
            vector=test_vector
        )
        print(f"✅ Inserted test document with UUID: {uuid}")
        
        # Check count
        count = db_client.get_document_count()
        print(f"✅ Total documents in collection: {count}")
        
        # Test search
        print("\nTesting vector search...")
        results = db_client.search_similar_docs(test_vector, limit=1)
        if results:
            print(f"✅ Found {len(results)} result(s)")
            print(f"   Title: {results[0]['title']}")
            print(f"   Similarity: {results[0]['similarity']:.4f}")
        
        # Cleanup
        db_client.delete_all_documents()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        db_client.disconnect()
