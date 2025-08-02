"""
Storage Nodes Module

This module contains LangGraph nodes for storage operations using LangChain and ChromaDB.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path
from langgraph.graph import StateGraph, END
from .base_nodes import BaseNode
from ..processor.storage_manager import StorageManager
from ..state.storage_state import StorageNodeState

logger = logging.getLogger(__name__)


class StorageNode(BaseNode):
    """
    LangGraph node for storing chunks in vector database.
    
    This node stores embedded chunks in ChromaDB using LangChain.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the storage node.
        
        Args:
            config: Configuration dictionary containing storage settings
        """
        super().__init__(config)
        
        # Get LangChain storage settings
        langchain_settings = config.get('langchain_settings', {})
        chroma_settings = langchain_settings.get('chroma_settings', {})
        
        collection_name = chroma_settings.get('collection_name', 'pdf_documents')
        embedding_model = langchain_settings.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize storage manager
        self.storage_manager = StorageManager(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        logger.info(f"Storage node initialized with collection: {collection_name}")
    
    def process(self, state: StorageNodeState) -> StorageNodeState:
        """
        Process the state and store chunks in vector database.
        
        Args:
            state: Current state containing chunks to store
            
        Returns:
            Updated state with storage information
        """
        logger.info("Starting storage process")
        
        try:
            # Get chunks from state
            chunks = state.chunks
            if not chunks:
                logger.warning("No chunks to store")
                return state
            
            logger.info(f"Storing {len(chunks)} chunks in vector database")
            
            # Get document ID
            document_id = state.document_id if hasattr(state, 'document_id') and state.document_id else 'unknown'
            
            # Store chunks in vector database
            storage_info = self.storage_manager.store_chunks(chunks, document_id)
            
            # Get collection statistics
            collection_stats = self.storage_manager.get_collection_stats()
            
            # Update state with storage information
            updated_state = state.model_copy(update={
                'storage_info': storage_info,
                'collection_stats': collection_stats,
                'collection_name': self.storage_manager.collection_name,
                'stored_chunks': storage_info.get('stored_chunks', 0),
                'storage_success': storage_info.get('stored_chunks', 0) > 0
            })
            
            logger.info(f"Storage completed successfully")
            logger.info(f"  - Stored chunks: {storage_info.get('stored_chunks', 0)}")
            logger.info(f"  - Collection: {self.storage_manager.collection_name}")
            logger.info(f"  - Document ID: {document_id}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error during storage: {e}")
            # Return state with error information
            return state.model_copy(update={
                'storage_error': str(e),
                'storage_info': {
                    'error': str(e),
                    'stored_chunks': 0
                },
                'storage_success': False
            })
    
    def get_node_name(self) -> str:
        """Get the name of this node."""
        return "storage_node"


class SearchNode(BaseNode):
    """
    LangGraph node for searching chunks in vector database.
    
    This node provides search capabilities using LangChain and ChromaDB.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the search node.
        
        Args:
            config: Configuration dictionary containing search settings
        """
        super().__init__(config)
        
        # Get LangChain storage settings
        langchain_settings = config.get('langchain_settings', {})
        chroma_settings = langchain_settings.get('chroma_settings', {})
        
        collection_name = chroma_settings.get('collection_name', 'pdf_documents')
        embedding_model = langchain_settings.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize storage manager
        self.storage_manager = StorageManager(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        logger.info(f"Search node initialized with collection: {collection_name}")
    
    def process(self, state: StorageNodeState) -> StorageNodeState:
        """
        Process the state and search for chunks.
        
        Args:
            state: Current state containing search query
            
        Returns:
            Updated state with search results
        """
        logger.info("Starting search process")
        
        try:
            # Get search query from state
            search_query = state.search_query
            if not search_query:
                logger.warning("No search query provided")
                return state
            
            # Get search parameters
            n_results = state.n_results if hasattr(state, 'n_results') else 10
            filter_metadata = state.filter_metadata if hasattr(state, 'filter_metadata') else None
            
            logger.info(f"Searching for: '{search_query}' with {n_results} results")
            
            # Search for chunks
            search_results = self.storage_manager.search_chunks(
                query=search_query,
                n_results=n_results,
                filter_metadata=filter_metadata
            )
            
            # Update state with search results
            updated_state = state.model_copy(update={
                'search_results': search_results,
                'search_query': search_query,
                'n_results': n_results,
                'results_count': len(search_results),
                'search_success': len(search_results) > 0
            })
            
            logger.info(f"Search completed successfully")
            logger.info(f"  - Query: '{search_query}'")
            logger.info(f"  - Results found: {len(search_results)}")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            # Return state with error information
            return state.model_copy(update={
                'search_error': str(e),
                'search_results': [],
                'results_count': 0,
                'search_success': False
            })
    
    def get_node_name(self) -> str:
        """Get the name of this node."""
        return "search_node"


class StorageStatsNode(BaseNode):
    """
    LangGraph node for getting storage statistics.
    
    This node provides statistics about the vector database.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the storage stats node.
        
        Args:
            config: Configuration dictionary containing storage settings
        """
        super().__init__(config)
        
        # Get LangChain storage settings
        langchain_settings = config.get('langchain_settings', {})
        chroma_settings = langchain_settings.get('chroma_settings', {})
        
        collection_name = chroma_settings.get('collection_name', 'pdf_documents')
        embedding_model = langchain_settings.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize storage manager
        self.storage_manager = StorageManager(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        logger.info(f"Storage stats node initialized with collection: {collection_name}")
    
    def process(self, state: StorageNodeState) -> StorageNodeState:
        """
        Process the state and get storage statistics.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with storage statistics
        """
        logger.info("Getting storage statistics")
        
        try:
            # Get comprehensive storage statistics
            storage_stats = self.storage_manager.get_storage_stats()
            
            # Update state with storage statistics
            updated_state = state.model_copy(update={
                'storage_stats': storage_stats,
                'collection_stats': storage_stats.get('collection_stats', {}),
                'file_system_stats': storage_stats.get('file_system', {}),
                'storage_manager_stats': storage_stats.get('storage_manager', {})
            })
            
            logger.info(f"Storage statistics retrieved successfully")
            logger.info(f"  - Collection: {storage_stats.get('collection_stats', {}).get('collection_name', 'unknown')}")
            logger.info(f"  - Total chunks: {storage_stats.get('collection_stats', {}).get('total_chunks', 0)}")
            logger.info(f"  - File size: {storage_stats.get('file_system', {}).get('total_size_bytes', 0)} bytes")
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            # Return state with error information
            return state.model_copy(update={
                'storage_error': str(e),
                'storage_stats': {
                    'error': str(e)
                }
            })
    
    def get_node_name(self) -> str:
        """Get the name of this node."""
        return "storage_stats_node"


def create_storage_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create a LangGraph for storage operations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StateGraph for storage operations
    """
    # Create the graph
    workflow = StateGraph(StorageNodeState)
    
    # Add nodes
    storage_node = StorageNode(config)
    workflow.add_node("store_chunks", storage_node.process)
    
    # Set entry point
    workflow.set_entry_point("store_chunks")
    
    # Set end point
    workflow.add_edge("store_chunks", END)
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("Storage graph created successfully")
    return app


def create_search_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create a LangGraph for search operations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StateGraph for search operations
    """
    # Create the graph
    workflow = StateGraph(StorageNodeState)
    
    # Add nodes
    search_node = SearchNode(config)
    workflow.add_node("search_chunks", search_node.process)
    
    # Set entry point
    workflow.set_entry_point("search_chunks")
    
    # Set end point
    workflow.add_edge("search_chunks", END)
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("Search graph created successfully")
    return app


def create_storage_stats_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create a LangGraph for storage statistics.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        StateGraph for storage statistics
    """
    # Create the graph
    workflow = StateGraph(StorageNodeState)
    
    # Add nodes
    storage_stats_node = StorageStatsNode(config)
    workflow.add_node("get_storage_stats", storage_stats_node.process)
    
    # Set entry point
    workflow.set_entry_point("get_storage_stats")
    
    # Set end point
    workflow.add_edge("get_storage_stats", END)
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("Storage stats graph created successfully")
    return app 