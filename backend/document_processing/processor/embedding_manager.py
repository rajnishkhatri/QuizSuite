"""
Embedding Manager Module

This module handles the generation of embeddings for different content types
using LangChain and LangGraph.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation for multimodal content using LangChain.
    
    Handles embeddings for:
    - Text chunks
    - Image descriptions (OCR text)
    - Table content
    - Code chunks
    - Figure descriptions
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 device_type: Optional[str] = None,
                 batch_size: int = 32):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the HuggingFace embedding model
            device: Device to use for computation ('cpu', 'cuda', 'mps', etc.)
            device_type: Device type for Mac support ('mps')
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.device_type = device_type
        self.device = device or self._get_optimal_device(device_type)
        self.batch_size = batch_size
        
        # Initialize the LangChain embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'batch_size': self.batch_size}
        )
        
        # Get embedding dimensions
        self.embedding_dim = self.embeddings.client.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Using device: {self.device}")
    
    def _get_optimal_device(self, device_type: Optional[str] = None) -> str:
        """
        Get the optimal device for embedding generation.
        
        Args:
            device_type: Preferred device type
            
        Returns:
            Optimal device string
        """
        # If device_type is specified, use it
        if device_type:
            if device_type == "mps" and self._is_mps_available():
                return "mps"
            elif device_type == "cuda" and self._is_cuda_available():
                return "cuda"
            elif device_type == "cpu":
                return "cpu"
        
        # Auto-detect optimal device
        if self._is_cuda_available():
            return "cuda"
        elif self._is_mps_available():
            return "mps"
        else:
            return "cpu"
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _is_mps_available(self) -> bool:
        """Check if MPS (Metal Performance Shaders) is available for Mac."""
        try:
            import torch
            return torch.backends.mps.is_available()
        except ImportError:
            return False
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings added
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        embedded_chunks = []
        
        # Group chunks by type for batch processing
        text_chunks = []
        image_chunks = []
        table_chunks = []
        code_chunks = []
        figure_chunks = []
        
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            if chunk_type == 'text':
                text_chunks.append(chunk)
            elif chunk_type == 'image':
                image_chunks.append(chunk)
            elif chunk_type == 'table':
                table_chunks.append(chunk)
            elif chunk_type == 'code':
                code_chunks.append(chunk)
            elif chunk_type == 'figure':
                figure_chunks.append(chunk)
        
        # Generate embeddings for each type
        logger.info(f"Processing {len(text_chunks)} text chunks")
        embedded_chunks.extend(self._embed_text_chunks(text_chunks))
        
        logger.info(f"Processing {len(image_chunks)} image chunks")
        embedded_chunks.extend(self._embed_image_chunks(image_chunks))
        
        logger.info(f"Processing {len(table_chunks)} table chunks")
        embedded_chunks.extend(self._embed_table_chunks(table_chunks))
        
        logger.info(f"Processing {len(code_chunks)} code chunks")
        embedded_chunks.extend(self._embed_code_chunks(code_chunks))
        
        logger.info(f"Processing {len(figure_chunks)} figure chunks")
        embedded_chunks.extend(self._embed_figure_chunks(figure_chunks))
        
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def _embed_text_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of text chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract text content
        texts = [chunk.get('content', '') for chunk in chunks]
        
        # Generate embeddings using LangChain
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embeddings[i]
            embedded_chunk['embedding_metadata'] = {
                'model': self.model_name,
                'dimension': self.embedding_dim,
                'device': self.device,
                'embedding_type': 'text'
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def _embed_image_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for image chunks using OCR text.
        
        Args:
            chunks: List of image chunks
            
        Returns:
            List of image chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract OCR text from images
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '')
            if content:
                texts.append(content)
            else:
                # Fallback to image metadata if no OCR text
                metadata = chunk.get('metadata', {})
                width = metadata.get('width', 0)
                height = metadata.get('height', 0)
                texts.append(f"Image with dimensions {width}x{height}")
        
        # Generate embeddings using LangChain
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embeddings[i]
            embedded_chunk['embedding_metadata'] = {
                'model': self.model_name,
                'dimension': self.embedding_dim,
                'device': self.device,
                'embedding_type': 'image_ocr'
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def _embed_table_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for table chunks.
        
        Args:
            chunks: List of table chunks
            
        Returns:
            List of table chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract table content
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '')
            if content:
                texts.append(content)
            else:
                # Fallback to table metadata
                metadata = chunk.get('metadata', {})
                rows = metadata.get('table_rows', 0)
                cols = metadata.get('table_columns', 0)
                texts.append(f"Table with {rows} rows and {cols} columns")
        
        # Generate embeddings using LangChain
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embeddings[i]
            embedded_chunk['embedding_metadata'] = {
                'model': self.model_name,
                'dimension': self.embedding_dim,
                'device': self.device,
                'embedding_type': 'table'
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def _embed_code_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for code chunks.
        
        Args:
            chunks: List of code chunks
            
        Returns:
            List of code chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract code content
        texts = [chunk.get('content', '') for chunk in chunks]
        
        # Generate embeddings using LangChain
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embeddings[i]
            embedded_chunk['embedding_metadata'] = {
                'model': self.model_name,
                'dimension': self.embedding_dim,
                'device': self.device,
                'embedding_type': 'code'
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def _embed_figure_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for figure chunks.
        
        Args:
            chunks: List of figure chunks
            
        Returns:
            List of figure chunks with embeddings
        """
        if not chunks:
            return []
        
        # Extract figure content (OCR text or metadata)
        texts = []
        for chunk in chunks:
            content = chunk.get('content', '')
            if content:
                texts.append(content)
            else:
                # Fallback to figure metadata
                metadata = chunk.get('metadata', {})
                fig_type = metadata.get('figure_type', 'unknown')
                width = metadata.get('width', 0)
                height = metadata.get('height', 0)
                texts.append(f"{fig_type} figure with dimensions {width}x{height}")
        
        # Generate embeddings using LangChain
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = chunk.copy()
            embedded_chunk['embedding'] = embeddings[i]
            embedded_chunk['embedding_metadata'] = {
                'model': self.model_name,
                'dimension': self.embedding_dim,
                'device': self.device,
                'embedding_type': 'figure'
            }
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def get_embedding_stats(self, embedded_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the generated embeddings.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            
        Returns:
            Dictionary with embedding statistics
        """
        stats = {
            'total_chunks': len(embedded_chunks),
            'embedding_dimension': self.embedding_dim,
            'model_used': self.model_name,
            'device_used': self.device,
            'device_type': self.device_type,
            'embeddings_by_type': {},
            'embedding_quality': {}
        }
        
        # Count embeddings by type
        for chunk in embedded_chunks:
            embedding_metadata = chunk.get('embedding_metadata', {})
            embedding_type = embedding_metadata.get('embedding_type', 'unknown')
            stats['embeddings_by_type'][embedding_type] = stats['embeddings_by_type'].get(embedding_type, 0) + 1
        
        # Calculate embedding quality metrics
        embedding_lengths = []
        for chunk in embedded_chunks:
            embedding = chunk.get('embedding', [])
            if embedding:
                embedding_lengths.append(len(embedding))
        
        if embedding_lengths:
            stats['embedding_quality'] = {
                'average_length': np.mean(embedding_lengths),
                'min_length': np.min(embedding_lengths),
                'max_length': np.max(embedding_lengths),
                'std_length': np.std(embedding_lengths)
            }
        
        return stats
    
    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        """
        Save embeddings to files.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            output_dir: Directory to save embeddings
            
        Returns:
            Dictionary with save information
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings by type
        embeddings_by_type = {}
        for chunk in embedded_chunks:
            embedding_type = chunk.get('embedding_metadata', {}).get('embedding_type', 'unknown')
            if embedding_type not in embeddings_by_type:
                embeddings_by_type[embedding_type] = []
            embeddings_by_type[embedding_type].append(chunk)
        
        saved_files = {}
        
        for embedding_type, chunks in embeddings_by_type.items():
            # Save embeddings as numpy arrays
            embeddings_file = output_dir / f"{embedding_type}_embeddings.npy"
            embeddings = np.array([chunk['embedding'] for chunk in chunks])
            np.save(embeddings_file, embeddings)
            
            # Save metadata
            metadata_file = output_dir / f"{embedding_type}_metadata.json"
            metadata = {
                'embedding_type': embedding_type,
                'count': len(chunks),
                'dimension': self.embedding_dim,
                'model': self.model_name,
                'device': self.device,
                'device_type': self.device_type,
                'chunk_ids': [chunk.get('chunk_index', i) for i, chunk in enumerate(chunks)]
            }
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_files[embedding_type] = {
                'embeddings_file': str(embeddings_file),
                'metadata_file': str(metadata_file),
                'count': len(chunks)
            }
        
        logger.info(f"Saved embeddings to {output_dir}")
        return saved_files 