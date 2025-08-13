# 🌸 FAISS Search for Automatic Flower Classification
# Author: Imad SANOUSSI
# GitHub: https://github.com/ImadSANOUSSI

"""
FAISS Search module for similarity search and feature matching.

This module provides:
- FAISS index creation and management
- Fast similarity search
- Feature vector indexing
- Nearest neighbor search
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pickle

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)

class FlowerFAISSSearch:
    """
    FAISS-based similarity search for flower features.
    
    Provides fast similarity search using:
    - IVFFlat index for large datasets
    - L2 distance metric
    - GPU acceleration support (if available)
    """
    
    def __init__(self, 
                 index_type: str = "IVFFlat",
                 dimension: int = 1280,
                 nlist: int = 100,
                 nprobe: int = 10,
                 metric: str = "L2",
                 use_gpu: bool = False):
        """
        Initialize FAISS search engine.
        
        Args:
            index_type: Type of FAISS index
            dimension: Dimension of feature vectors
            nlist: Number of clusters for IVFFlat
            nprobe: Number of clusters to visit during search
            metric: Distance metric (L2, IP, etc.)
            use_gpu: Whether to use GPU acceleration
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu")
        
        self.index_type = index_type
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric
        self.use_gpu = use_gpu
        
        self.index = None
        self.feature_database = {}
        self.is_trained = False
        
        # Initialize FAISS index
        self._create_index()
        
        logger.info(f"🔍 FAISS Search initialized: {index_type}")
        logger.info(f"📐 Feature dimension: {dimension}")
        logger.info(f"🎯 Number of clusters: {nlist}")
    
    def _create_index(self):
        """Create and configure FAISS index."""
        try:
            if self.index_type == "IVFFlat":
                # Create quantizer
                quantizer = faiss.IndexFlatL2(self.dimension)
                
                # Create IVFFlat index
                self.index = faiss.IndexIVFFlat(
                    quantizer, 
                    self.dimension, 
                    self.nlist, 
                    faiss.METRIC_L2
                )
                
            elif self.index_type == "Flat":
                # Simple flat index for small datasets
                self.index = faiss.IndexFlatL2(self.dimension)
                
            elif self.index_type == "LSH":
                # Locality Sensitive Hashing
                self.index = faiss.IndexLSH(self.dimension, self.dimension // 8)
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Configure search parameters
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            logger.info(f"✅ FAISS index created: {self.index_type}")
            
        except Exception as e:
            logger.error(f"❌ Error creating FAISS index: {str(e)}")
            raise
    
    def add_features(self, 
                    features: np.ndarray, 
                    metadata: List[Dict[str, Any]]):
        """
        Add feature vectors to the index.
        
        Args:
            features: Feature vectors array (N x dimension)
            metadata: List of metadata for each feature
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized")
        
        try:
            # Ensure features are float32
            features = features.astype(np.float32)
            
            # Add features to index
            if self.index_type == "IVFFlat" and not self.is_trained:
                # Train the index first
                self.index.train(features)
                self.is_trained = True
                logger.info("🎯 Index trained with feature vectors")
            
            # Add vectors to index
            self.index.add(features)
            
            # Store metadata
            for i, meta in enumerate(metadata):
                self.feature_database[i] = meta
            
            logger.info(f"📥 Added {len(features)} feature vectors to index")
            logger.info(f"📊 Total vectors in index: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"❌ Error adding features: {str(e)}")
            raise
    
    def search(self, 
               query_features: np.ndarray, 
               k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar features.
        
        Args:
            query_features: Query feature vector
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized")
        
        try:
            # Ensure query features are float32
            query_features = query_features.astype(np.float32)
            
            # Reshape if needed
            if query_features.ndim == 1:
                query_features = query_features.reshape(1, -1)
            
            # Perform search
            distances, indices = self.index.search(query_features, k)
            
            logger.info(f"🔍 Search completed: {k} nearest neighbors found")
            
            return distances[0], indices[0]
            
        except Exception as e:
            logger.error(f"❌ Error during search: {str(e)}")
            raise
    
    def search_with_metadata(self, 
                            query_features: np.ndarray, 
                            k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar features with metadata.
        
        Args:
            query_features: Query feature vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of results with metadata
        """
        try:
            distances, indices = self.search(query_features, k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx != -1:  # Valid index
                    result = {
                        "rank": i + 1,
                        "distance": float(distance),
                        "index": int(idx),
                        "metadata": self.feature_database.get(idx, {})
                    }
                    results.append(result)
            
            logger.info(f"📋 Search results with metadata: {len(results)} items")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during metadata search: {str(e)}")
            raise
    
    def batch_search(self, 
                     query_features: np.ndarray, 
                     k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform batch search for multiple queries.
        
        Args:
            query_features: Batch of query feature vectors (N x dimension)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (distances, indices) for all queries
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized")
        
        try:
            # Ensure features are float32
            query_features = query_features.astype(np.float32)
            
            # Perform batch search
            distances, indices = self.index.search(query_features, k)
            
            logger.info(f"🔍 Batch search completed: {len(query_features)} queries")
            
            return distances, indices
            
        except Exception as e:
            logger.error(f"❌ Error during batch search: {str(e)}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index."""
        if self.index is None:
            return {"error": "Index not initialized"}
        
        try:
            stats = {
                "index_type": self.index_type,
                "dimension": self.dimension,
                "total_vectors": self.index.ntotal,
                "is_trained": self.is_trained,
                "nlist": self.nlist,
                "nprobe": self.nprobe
            }
            
            # Add index-specific stats
            if hasattr(self.index, 'nprobe'):
                stats["current_nprobe"] = self.index.nprobe
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting index stats: {str(e)}")
            return {"error": str(e)}
    
    def save_index(self, index_path: str):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index not initialized")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = index_path.replace('.index', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.feature_database, f)
            
            logger.info(f"💾 Index saved to: {index_path}")
            logger.info(f"💾 Metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving index: {str(e)}")
            raise
    
    def load_index(self, index_path: str):
        """Load a FAISS index from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = index_path.replace('.index', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.feature_database = pickle.load(f)
            
            # Update state
            self.is_trained = True
            self.dimension = self.index.d
            
            logger.info(f"📂 Index loaded from: {index_path}")
            logger.info(f"📊 Index contains {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"❌ Error loading index: {str(e)}")
            raise
    
    def clear_index(self):
        """Clear all vectors from the index."""
        if self.index is None:
            return
        
        try:
            # Reset index
            if hasattr(self.index, 'reset'):
                self.index.reset()
            else:
                # Create new index
                self._create_index()
            
            # Clear metadata
            self.feature_database.clear()
            self.is_trained = False
            
            logger.info("🧹 Index cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing index: {str(e)}")
            raise


def create_faiss_search(index_type: str = "IVFFlat", **kwargs) -> FlowerFAISSSearch:
    """
    Factory function to create a FAISS search engine.
    
    Args:
        index_type: Type of FAISS index
        **kwargs: Additional arguments for FlowerFAISSSearch
        
    Returns:
        Initialized FlowerFAISSSearch instance
    """
    return FlowerFAISSSearch(index_type=index_type, **kwargs)


# Example usage
if __name__ == "__main__":
    if FAISS_AVAILABLE:
        # Initialize FAISS search
        faiss_search = create_faiss_search(
            index_type="IVFFlat",
            dimension=1280,
            nlist=100
        )
        
        # Print index stats
        print("📊 Index Statistics:")
        print(faiss_search.get_index_stats())
        
        print("\n✅ FAISS Search module ready!")
    else:
        print("❌ FAISS not available. Install with: pip install faiss-cpu")
