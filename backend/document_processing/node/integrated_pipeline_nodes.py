"""
Integrated Pipeline Nodes

This module implements a complete integrated pipeline that includes:
1. Document ingestion
2. Content extraction and chunking
3. Embedding generation
4. Vector database storage
5. End state with comprehensive results
"""

import logging
from typing import Dict, Any, List
from pathlib import Path
import time

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from .base_nodes import TraceableNode
from ..processor.document_processor import DocumentProcessor
from ..processor.embedding_manager import EmbeddingManager
from ..processor.storage_manager import StorageManager
from ..state.unified_state import UnifiedPipelineState

logger = logging.getLogger(__name__)


class IntegratedIngestNode(TraceableNode):
    """Integrated document ingestion node."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_node_name(self) -> str:
        return "integrated_ingest_node"
    
    @traceable(run_type="chain", name="Document Ingestion Node")
    def process(self, state: UnifiedPipelineState) -> UnifiedPipelineState:
        """
        Ingest documents from configuration.
        
        Args:
            state: Current unified state
            
        Returns:
            Updated state with ingested documents
        """
        logger.info("Starting document ingestion")
        
        try:
            documents = self._load_documents_from_config()
            
            return state.model_copy(update={
                'documents': documents,
                'total_documents': len(documents),
                'success': True,
                'node_name': 'integrated_ingest_node'
            })
            
        except Exception as e:
            logger.error(f"Error in ingest node: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'integrated_ingest_node'
            })
    
    def _load_documents_from_config(self) -> List[Dict[str, Any]]:
        """Load documents from configuration."""
        documents = []
        
        # Get document categories from config
        categories = self.config.get('categories', [])
        
        for category in categories:
            doc_paths = category.get('doc_paths', [])
            category_name = category.get('name', 'unknown')
            
            for doc_path in doc_paths:
                if Path(doc_path).exists():
                    # Create simple document dictionary
                    document = {
                        'path': doc_path,
                        'file_path': doc_path,
                        'document_id': f"doc_{Path(doc_path).stem}_{int(time.time())}",
                        'document_type': 'pdf',
                        'content': f"Document from {category_name} category",
                        'processing_status': 'pending',
                        'metadata': {'category': category_name}
                    }
                    documents.append(document)
                    logger.info(f"Loaded document: {doc_path}")
                else:
                    logger.warning(f"Document not found: {doc_path}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents


class IntegratedProcessNode(TraceableNode):
    """Integrated document processing node."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get collection name from config
        langchain_settings = config.get('langchain_settings', {})
        chroma_settings = langchain_settings.get('chroma_settings', {})
        collection_name = chroma_settings.get('collection_name', 'pdf_documents')
        
        # Create document processor with correct collection name
        self.processor = DocumentProcessor(
            config=config,
            collection_name=collection_name
        )
    
    def get_node_name(self) -> str:
        return "integrated_process_node"
    
    @traceable(run_type="chain", name="Document Processing Node")
    def process(self, state: UnifiedPipelineState) -> UnifiedPipelineState:
        """
        Process documents with extraction and chunking.
        
        Args:
            state: Current unified state
            
        Returns:
            Updated state with processed chunks
        """
        logger.info("Starting document processing")
        
        try:
            documents = state.documents
            if not documents:
                logger.warning("No documents to process")
                return state.model_copy(update={
                    'error': "No documents to process",
                    'success': False,
                    'node_name': 'integrated_process_node'
                })
            
            all_chunks = []
            processed_documents = []
            
            for doc in documents:
                doc_path = doc.get('path')
                doc_id = doc.get('document_id')
                
                if not doc_path or not Path(doc_path).exists():
                    logger.warning(f"Document path not found: {doc_path}")
                    continue
                
                try:
                    # Process the document using DocumentProcessor
                    doc_dict = {
                        'path': doc_path,
                        'document_id': doc_id,
                        'type': 'pdf',
                        'content': f"Document from {doc.get('metadata', {}).get('category', 'unknown')} category"
                    }
                    processed_result = self.processor.process_document(doc_dict)
                    
                    if processed_result and 'chunks' in processed_result:
                        chunks = processed_result['chunks']
                        all_chunks.extend(chunks)
                        processed_documents.append(doc)
                        logger.info(f"Processed document {doc_id}: {len(chunks)} chunks")
                    else:
                        logger.warning(f"No chunks generated for document {doc_id}")
                        
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
                    continue
            
            # Create processing summary
            processing_summary = {
                'total_documents_processed': len(processed_documents),
                'total_chunks_created': len(all_chunks),
                'processing_success': True
            }
            
            return state.model_copy(update={
                'processed_documents': processed_documents,
                'chunks': all_chunks,
                'processing_summary': processing_summary,
                'success': True,
                'node_name': 'integrated_process_node'
            })
            
        except Exception as e:
            logger.error(f"Error in process node: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'integrated_process_node'
            })


class IntegratedEmbeddingNode(TraceableNode):
    """Integrated embedding generation node."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        langchain_settings = config.get('langchain_settings', {})
        self.embedding_manager = EmbeddingManager(
            model_name=langchain_settings.get('embedding_model', 'all-MiniLM-L6-v2'),
            device=langchain_settings.get('device_type', 'cpu')
        )
    
    def get_node_name(self) -> str:
        return "integrated_embedding_node"
    
    @traceable(run_type="chain", name="Embedding Generation Node")
    def process(self, state: UnifiedPipelineState) -> UnifiedPipelineState:
        """
        Generate embeddings for document chunks.
        
        Args:
            state: Current unified state
            
        Returns:
            Updated state with embedded chunks
        """
        logger.info("Starting embedding generation")
        
        try:
            chunks = state.chunks
            if not chunks:
                logger.warning("No chunks to embed")
                return state.model_copy(update={
                    'error': "No chunks to embed",
                    'success': False,
                    'node_name': 'integrated_embedding_node'
                })
            
            embedded_chunks = []
            embedding_stats = {}
            
            # Generate embeddings for all chunks at once
            try:
                embedded_chunks = self.embedding_manager.generate_embeddings(chunks)
                logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                return state.model_copy(update={
                    'error': f"Embedding generation failed: {e}",
                    'success': False,
                    'node_name': 'integrated_embedding_node'
                })
            
            # Create embedding statistics
            embedding_stats = {
                'total_chunks_processed': len(chunks),
                'successfully_embedded': len(embedded_chunks),
                'embedding_model': self.embedding_manager.model_name,
                'embedding_dimension': len(embedded_chunks[0]['embedding']) if embedded_chunks else 0
            }
            
            return state.model_copy(update={
                'embedded_chunks': embedded_chunks,
                'embedding_stats': embedding_stats,
                'embedding_model': self.embedding_manager.model_name,
                'success': True,
                'node_name': 'integrated_embedding_node'
            })
            
        except Exception as e:
            logger.error(f"Error in embedding node: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'integrated_embedding_node'
            })


class IntegratedStorageNode(TraceableNode):
    """Integrated storage node."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
    
    def get_node_name(self) -> str:
        return "integrated_storage_node"
    
    def _get_storage_manager(self) -> StorageManager:
        """Get or create storage manager with config collection name."""
        langchain_settings = self.config.get('langchain_settings', {})
        chroma_settings = langchain_settings.get('chroma_settings', {})
        
        # Use collection name from config
        collection_name = chroma_settings.get('collection_name', 'pdf_documents')
        persist_directory = chroma_settings.get('persist_directory', 'storage/chroma_db_pdf')
        
        return StorageManager(
            persist_directory=Path(persist_directory),
            collection_name=collection_name,
            embedding_model=langchain_settings.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
    
    @traceable(run_type="chain", name="Storage Node")
    def process(self, state: UnifiedPipelineState) -> UnifiedPipelineState:
        """
        Store embedded chunks in vector database.
        
        Args:
            state: Current unified state
            
        Returns:
            Updated state with storage results
        """
        logger.info("Starting storage operations")
        
        try:
            embedded_chunks = state.embedded_chunks
            if not embedded_chunks:
                logger.warning("No embedded chunks to store")
                return state.model_copy(update={
                    'error': "No embedded chunks to store",
                    'success': False,
                    'node_name': 'integrated_storage_node'
                })
            
            stored_chunks = []
            collection_stats = {}
            
            # Group chunks by document for batch storage
            document_chunks = {}
            for chunk in embedded_chunks:
                doc_id = chunk.get('document_id', 'unknown')
                if doc_id not in document_chunks:
                    document_chunks[doc_id] = []
                document_chunks[doc_id].append(chunk)
            
            # Store chunks for each document
            for doc_id, chunks in document_chunks.items():
                try:
                    storage_result = self._get_storage_manager().store_chunks(chunks, doc_id)
                    if storage_result.get('stored_chunks', 0) > 0:
                        stored_chunks.extend(chunks)
                        logger.info(f"Stored {storage_result['stored_chunks']} chunks for document {doc_id}")
                    else:
                        logger.warning(f"No chunks stored for document {doc_id}")
                        
                except Exception as e:
                    logger.error(f"Error storing chunks for document {doc_id}: {e}")
                    continue
            
            # Get collection statistics
            collection_stats = self._get_storage_manager().get_collection_stats()
            
            return state.model_copy(update={
                'stored_chunks': stored_chunks,
                'collection_stats': collection_stats,
                'storage_success': len(stored_chunks) > 0,
                'success': True,
                'node_name': 'integrated_storage_node'
            })
            
        except Exception as e:
            logger.error(f"Error in storage node: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'integrated_storage_node'
            })


class IntegratedSummaryNode(TraceableNode):
    """Integrated summary node that creates final pipeline results."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def get_node_name(self) -> str:
        return "integrated_summary_node"
    
    @traceable(run_type="chain", name="Summary Node")
    def process(self, state: UnifiedPipelineState) -> UnifiedPipelineState:
        """
        Create final summary state with comprehensive results.
        
        Args:
            state: Current unified state
            
        Returns:
            Final state with comprehensive summary
        """
        logger.info("Creating final summary state")
        
        try:
            # Create comprehensive summary
            final_summary = {
                'pipeline_success': True,
                'total_documents_processed': len(state.processed_documents),
                'total_chunks_created': len(state.chunks),
                'total_chunks_embedded': len(state.embedded_chunks),
                'total_chunks_stored': len(state.stored_chunks),
                'embedding_stats': state.embedding_stats,
                'storage_stats': state.collection_stats,
                'processing_summary': state.processing_summary,
                'pipeline_completion_time': time.time()
            }
            
            return state.model_copy(update={
                'final_summary': final_summary,
                'pipeline_success': True,
                'success': True,
                'node_name': 'integrated_summary_node'
            })
            
        except Exception as e:
            logger.error(f"Error in summary node: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'integrated_summary_node'
            })


def create_integrated_pipeline_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create the complete integrated pipeline graph.
    
    Flow: START → Ingest → Process → Embed → Store → Summary → END
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled StateGraph
    """
    # Create the graph with UnifiedPipelineState as base
    workflow = StateGraph(UnifiedPipelineState)
    
    # Add nodes
    ingest_node = IntegratedIngestNode(config)
    process_node = IntegratedProcessNode(config)
    embed_node = IntegratedEmbeddingNode(config)
    storage_node = IntegratedStorageNode(config)
    summary_node = IntegratedSummaryNode(config)
    
    # Add all nodes
    workflow.add_node("ingest", ingest_node.process)
    workflow.add_node("process", process_node.process)
    workflow.add_node("embed", embed_node.process)
    workflow.add_node("store", storage_node.process)
    workflow.add_node("summary", summary_node.process)
    
    # Define the complete flow
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "process")
    workflow.add_edge("process", "embed")
    workflow.add_edge("embed", "store")
    workflow.add_edge("store", "summary")
    workflow.add_edge("summary", END)
    
    return workflow.compile() 