#!/usr/bin/env python3
"""
Qdrant Vector Database Manager

This module handles all vector database operations for face embeddings using Qdrant.
It provides a clean interface for storing, searching, and managing face embeddings.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct


class QdrantManager:
    """
    Manages face embeddings using Qdrant vector database.
    Handles all vector operations including add, search, delete, and metadata management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant client and create collection if needed
        
        Args:
            config: Configuration dictionary containing vector database settings
        """
        self.config = config.get('vector_database', {})
        self.collection_name = self.config.get('collection_name', 'face_embeddings')
        self.vector_size = self.config.get('vector_size', 512)
        self.distance_metric = self.config.get('distance_metric', 'Cosine')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize Qdrant client
        try:
            if self.config.get('mode') == 'memory':
                # Use in-memory mode
                self.client = QdrantClient(":memory:")
                self.logger.info("Initialized Qdrant client in memory mode")
            else:
                # Use persistent mode (default to localhost)
                host = self.config.get('host', 'localhost')
                port = self.config.get('port', 6333)
                self.client = QdrantClient(host=host, port=port)
                self.logger.info(f"Initialized Qdrant client connecting to {host}:{port}")
            
            # Create collection if it doesn't exist
            self._create_collection()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def _create_collection(self):
        """Create the face embeddings collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self._get_distance_metric()
                    )
                )
                self.logger.info(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")
            else:
                self.logger.info(f"Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            raise
    
    def _get_distance_metric(self) -> Distance:
        """Convert string distance metric to Qdrant Distance enum"""
        distance_map = {
            'Cosine': Distance.COSINE,
            'Euclidean': Distance.EUCLID,
            'Dot': Distance.DOT
        }
        return distance_map.get(self.distance_metric, Distance.COSINE)
    
    def add_embedding(self, person_id: int, embedding: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """
        Add a face embedding to the vector database
        
        Args:
            person_id: Unique person identifier
            embedding: Face embedding vector (numpy array)
            metadata: Additional metadata (name, quality, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert numpy array to list
            if isinstance(embedding, np.ndarray):
                vector = embedding.tolist()
            else:
                vector = list(embedding)
            
            # Ensure vector has correct size
            if len(vector) != self.vector_size:
                self.logger.error(f"Vector size mismatch: expected {self.vector_size}, got {len(vector)}")
                return False
            
            # Create point with person_id as point_id
            point = PointStruct(
                id=person_id,
                vector=vector,
                payload={
                    'person_id': person_id,
                    **metadata
                }
            )
            
            # Insert point
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            self.logger.info(f"Added embedding for person {person_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add embedding for person {person_id}: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, 
                      threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar face embeddings
        
        Args:
            query_embedding: Query face embedding vector
            k: Number of top results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar faces with metadata and scores
        """
        try:
            # Convert numpy array to list
            if isinstance(query_embedding, np.ndarray):
                query_vector = query_embedding.tolist()
            else:
                query_vector = list(query_embedding)
            
            # Ensure vector has correct size
            if len(query_vector) != self.vector_size:
                self.logger.error(f"Query vector size mismatch: expected {self.vector_size}, got {len(query_vector)}")
                return []
            
            # Search for similar vectors
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                score_threshold=threshold,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'person_id': result.payload.get('person_id', result.id),
                    'name': result.payload.get('name', 'Unknown'),
                    'similarity': float(result.score),
                    'quality': result.payload.get('quality', 0.0),
                    'metadata': result.payload
                })
            
            self.logger.debug(f"Found {len(results)} similar faces")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar faces: {e}")
            return []
    
    def delete_embedding(self, person_id: int) -> bool:
        """
        Delete a face embedding from the vector database
        
        Args:
            person_id: Person identifier to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete point by ID
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[person_id])
            )
            
            self.logger.info(f"Deleted embedding for person {person_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete embedding for person {person_id}: {e}")
            return False
    
    def get_embedding_count(self) -> int:
        """
        Get the total number of embeddings stored
        
        Returns:
            Number of embeddings in the collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            self.logger.error(f"Failed to get embedding count: {e}")
            return 0
    
    def get_embedding(self, person_id: int) -> Optional[np.ndarray]:
        """
        Get a specific embedding by person ID
        
        Args:
            person_id: Person identifier
            
        Returns:
            Face embedding vector or None if not found
        """
        try:
            # Retrieve point by ID
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[person_id],
                with_vectors=True
            )
            
            if points and len(points) > 0:
                return np.array(points[0].vector)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get embedding for person {person_id}: {e}")
            return None
    
    def update_embedding(self, person_id: int, embedding: np.ndarray, 
                        metadata: Dict[str, Any]) -> bool:
        """
        Update an existing face embedding
        
        Args:
            person_id: Person identifier
            embedding: New face embedding vector
            metadata: Updated metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Use upsert to update (same as add_embedding)
        return self.add_embedding(person_id, embedding, metadata)
    
    def clear_all(self) -> bool:
        """
        Clear all embeddings from the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            self.logger.info("Cleared all embeddings from collection")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear all embeddings: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': collection_info.config.params.vectors.size,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.name,
                'points_count': collection_info.points_count,
                'status': collection_info.status.name
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Test the QdrantManager
    import json
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize manager
    manager = QdrantManager(config)
    
    # Test adding an embedding
    test_embedding = np.random.rand(512).astype(np.float32)
    success = manager.add_embedding(
        person_id=1,
        embedding=test_embedding,
        metadata={'name': 'Test Person', 'quality': 0.95}
    )
    
    print(f"Add embedding: {'Success' if success else 'Failed'}")
    
    # Test searching
    results = manager.search_similar(test_embedding, k=5)
    print(f"Search results: {len(results)} found")
    
    # Test getting count
    count = manager.get_embedding_count()
    print(f"Total embeddings: {count}")
    
    # Test collection info
    info = manager.get_collection_info()
    print(f"Collection info: {info}")
