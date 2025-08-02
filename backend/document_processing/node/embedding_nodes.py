"""
Embedding Nodes Module

This module contains LangGraph nodes for embedding generation using LangChain.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langgraph.graph import StateGraph, END
from .base_nodes import BaseNode
from ..processor.embedding_manager import EmbeddingManager
from ..state.embed_state import EmbedNodeState

logger = logging.getLogger(__name__)


class EmbeddingNode(BaseNode):
    """
    LangGraph node for generating embeddings using LangChain.
    
    This node processes chunks and generates embeddings for different content types:
    - Text chunks
    - Image descriptions (OCR text)
    - Table content
    - Code chunks
    - Figure descriptions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding node.
        
        Args:
            config: Configuration dictionary containing embedding settings
        """
        super().__init__(config)
        
        # Get LangChain embedding settings
        langchain_settings = config.get('langchain_settings', {})
        embedding_model = langchain_settings.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        embedding_device = langchain_settings.get('embedding_device', 'cpu')
        device_type = langchain_settings.get('device_type', 'mps')
        embedding_batch_size = langchain_settings.get('embedding_batch_size', 32)
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            device=embedding_device,
            device_type=device_type,
            batch_size=embedding_batch_size
        )
        
        logger.info(f"Embedding node initialized with model: {embedding_model}")
        logger.info(f"Device: {embedding_device}, Device Type: {device_type}")
    
    def process(self, state: EmbedNodeState) -> EmbedNodeState:
        """
        Process the state and generate embeddings for chunks.
        
        Args:
            state: Current state containing chunks to embed
            
        Returns:
            Updated state with embedded chunks
        """
        logger.info("Starting embedding generation process")
        
        try:
            # Get chunks from state
            chunks = state.chunks
            if not chunks:
                logger.warning("No chunks to embed")
                return state
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Generate embeddings using LangChain
            embedded_chunks = self.embedding_manager.generate_embeddings(chunks)
            
            # Get embedding statistics
            embedding_stats = self.embedding_manager.get_embedding_stats(embedded_chunks)
            
            # Update state with embedded chunks and statistics
            updated_state = state.model_copy(update={
                'chunks': embedded_chunks,
                'embedding_stats': embedding_stats,
                'embedding_model': self.embedding_manager.model_name,
                'embedding_dimension': self.embedding_manager.embedding_dim,
                'embedding_device': self.embedding_manager.device
            })
            
            logger.info(f"Embedding generation completed successfully")
            logger.info(f"  - Total chunks: {len(embedded_chunks)}")
            logger.info(f"  - Embedding model: {self.embedding_manager.model_name}")
            logger.info(f"  - Embedding dimension: {self.embedding_manager.embedding_dim}")
            logger.info(f"  - Device: {self.embedding_manager.device}")
            
            # Log embedding statistics
            for embedding_type, count in embedding_stats.get('embeddings_by_type', {}).items():
                logger.info(f"  - {embedding_type} embeddings: {count}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            # Return state with error information
            return state.model_copy(update={
                'embedding_error': str(e),
                'embedding_stats': {
                    'error': str(e),
                    'total_chunks': len(state.chunks) if state.chunks else 0
                }
            })
    
    def get_node_name(self) -> str:
        """Get the name of this node."""
        return "embedding_node"


class EmbeddingWithStorageNode(BaseNode):
    """
    LangGraph node for generating embeddings and storing them.
    
    This node combines embedding generation with storage in a single step.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding with storage node.
        
        Args:
            config: Configuration dictionary containing embedding and storage settings
        """
        super().__init__(config)
        
        # Get LangChain settings
        langchain_settings = config.get('langchain_settings', {})
        embedding_model = langchain_settings.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        embedding_device = langchain_settings.get('embedding_device', 'cpu')
        device_type = langchain_settings.get('device_type', 'mps')
        embedding_batch_size = langchain_settings.get('embedding_batch_size', 32)
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            device=embedding_device,
            device_type=device_type,
            batch_size=embedding_batch_size
        )
        
        # Initialize storage manager
        from ..processor.storage_manager import StorageManager
        chroma_settings = langchain_settings.get('chroma_settings', {})
        collection_name = chroma_settings.get('collection_name', 'pdf_documents')
        
        self.storage_manager = StorageManager(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        logger.info(f"Embedding with storage node initialized")
        logger.info(f"  - Embedding model: {embedding_model}")
        logger.info(f"  - Device: {embedding_device}, Device Type: {device_type}")
        logger.info(f"  - Collection name: {collection_name}")
    
    def process(self, state: EmbedNodeState) -> EmbedNodeState:
        """
        Process the state, generate embeddings, and store them.
        
        Args:
            state: Current state containing chunks to embed and store
            
        Returns:
            Updated state with embedded chunks and storage information
        """
        logger.info("Starting embedding generation and storage process")
        
        try:
            # Get chunks from state
            chunks = state.chunks
            if not chunks:
                logger.warning("No chunks to embed and store")
                return state
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Generate embeddings using LangChain
            embedded_chunks = self.embedding_manager.generate_embeddings(chunks)
            
            # Get embedding statistics
            embedding_stats = self.embedding_manager.get_embedding_stats(embedded_chunks)
            
            # Store chunks in vector database
            document_id = state.document_id if hasattr(state, 'document_id') else 'unknown'
            storage_info = self.storage_manager.store_chunks(embedded_chunks, document_id)
            
            # Update state with embedded chunks, statistics, and storage info
            updated_state = state.model_copy(update={
                'chunks': embedded_chunks,
                'embedding_stats': embedding_stats,
                'storage_info': storage_info,
                'embedding_model': self.embedding_manager.model_name,
                'embedding_dimension': self.embedding_manager.embedding_dim,
                'embedding_device': self.embedding_manager.device,
                'collection_name': self.storage_manager.collection_name
            })
            
            logger.info(f"Embedding generation and storage completed successfully")
            logger.info(f"  - Total chunks: {len(embedded_chunks)}")
            logger.info(f"  - Stored chunks: {storage_info.get('stored_chunks', 0)}")
            logger.info(f"  - Collection: {self.storage_manager.collection_name}")
            
            # Log embedding statistics
            for embedding_type, count in embedding_stats.get('embeddings_by_type', {}).items():
                logger.info(f"  - {embedding_type} embeddings: {count}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error during embedding generation and storage: {e}")
            # Return state with error information
            return state.model_copy(update={
                'embedding_error': str(e),
                'embedding_stats': {
                    'error': str(e),
                    'total_chunks': len(state.chunks) if state.chunks else 0
                },
                'storage_info': {
                    'error': str(e),
                    'stored_chunks': 0
                }
            })
    
    def get_node_name(self) -> str:
        """Get the name of this node."""
        return "embedding_with_storage_node"


def create_embedding_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create a LangGraph for embedding generation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StateGraph for embedding generation
    """
    # Create the graph
    workflow = StateGraph(EmbedNodeState)
    
    # Add nodes
    embedding_node = EmbeddingNode(config)
    workflow.add_node("generate_embeddings", embedding_node.process)
    
    # Set entry point
    workflow.set_entry_point("generate_embeddings")
    
    # Set end point
    workflow.add_edge("generate_embeddings", END)
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("Embedding graph created successfully")
    return app


def create_embedding_with_storage_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create a LangGraph for embedding generation with storage.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StateGraph for embedding generation with storage
    """
    # Create the graph
    workflow = StateGraph(EmbedNodeState)
    
    # Add nodes
    embedding_with_storage_node = EmbeddingWithStorageNode(config)
    workflow.add_node("generate_embeddings_and_store", embedding_with_storage_node.process)
    
    # Set entry point
    workflow.set_entry_point("generate_embeddings_and_store")
    
    # Set end point
    workflow.add_edge("generate_embeddings_and_store", END)
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("Embedding with storage graph created successfully")
    return app 