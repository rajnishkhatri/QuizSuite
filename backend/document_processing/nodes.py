"""
LangGraph nodes for document processing pipeline.

This module implements the nodes for the document processing graph:
START → Ingest Documents → Process (Split, Add Metadata, Clean) → END
"""

import logging
from typing import Dict, Any, List
from pathlib import Path
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .models import (
    QuizConfig,
    ProcessedDocument,
    DocumentChunk,
    ModalityType
)
from .state.ingest_state import IngestNodeState
from .state.process_state import ProcessNodeState
from .state.end_state import EndNodeState
from .document_ingestor import DocumentIngestionManager
from .chunking_strategy import ModalityAwareChunkingStrategy
from .metadata_enricher import MetadataEnricher
from .text_cleaner import ModalityAwareTextCleaner


logger = logging.getLogger(__name__)


class DocumentProcessingNodes:
    """
    LangGraph nodes for document processing pipeline.
    
    Follows Single Responsibility Principle - each node has one clear purpose.
    """
    
    def __init__(self, config: QuizConfig):
        """Initialize nodes with configuration."""
        self.config = config
        self.ingestion_manager = DocumentIngestionManager()
        self.chunking_strategy = ModalityAwareChunkingStrategy()
        self.metadata_enricher = MetadataEnricher()
        self.text_cleaner = ModalityAwareTextCleaner()
    
    def ingest_documents_node(self, state: IngestNodeState) -> IngestNodeState:
        """
        Node for ingesting documents.
        
        This node handles the ingestion of documents from various sources.
        """
        try:
            logger.info("Starting document ingestion node")
            
            # Extract file paths from config categories
            file_paths = self._extract_file_paths_from_config()
            
            if not file_paths:
                error_msg = "No valid file paths found in configuration"
                logger.error(error_msg)
                state.errors.append(error_msg)
                return state
            
            # Ingest documents
            documents = []
            for file_path in file_paths:
                try:
                    if file_path.exists():
                        document = self._ingest_single_document(file_path)
                        documents.append(document)
                        logger.info(f"Successfully ingested: {file_path}")
                    else:
                        logger.warning(f"File not found: {file_path}")
                except Exception as e:
                    error_msg = f"Error ingesting {file_path}: {e}"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
            
            # Update state - convert ProcessedDocument objects to dictionaries for LangGraph serialization
            state.documents = [doc.model_dump() for doc in documents]
            state.file_paths = file_paths
            state.config = self.config
            state.messages.append({
                "type": "info",
                "content": f"Ingested {len(documents)} documents from {len(file_paths)} file paths"
            })
            
            logger.info(f"Document ingestion completed. Processed {len(documents)} documents")
            
        except Exception as e:
            error_msg = f"Fatal error in ingest node: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def process_documents_node(self, state: ProcessNodeState) -> ProcessNodeState:
        """
        Node for processing documents (chunking, cleaning, metadata enrichment).
        
        This node handles the complete processing pipeline for each document.
        """
        try:
            logger.info("Starting document processing node")
            
            # Get documents from previous node
            document_dicts = getattr(state, 'documents', [])
            logger.info(f"Retrieved {len(document_dicts)} documents from state")
            
            # Handle documents from previous node
            documents = []
            for doc_item in document_dicts:
                try:
                    if isinstance(doc_item, ProcessedDocument):
                        # Already a ProcessedDocument object
                        documents.append(doc_item)
                    elif isinstance(doc_item, dict):
                        # Dictionary that needs to be converted
                        document = ProcessedDocument(**doc_item)
                        documents.append(document)
                    else:
                        # Try to add it anyway if it's a ProcessedDocument-like object
                        if hasattr(doc_item, 'document_id') and hasattr(doc_item, 'file_path'):
                            documents.append(doc_item)
                        else:
                            logger.warning(f"Unexpected document type: {type(doc_item)}")
                except Exception as e:
                    logger.warning(f"Error handling document: {e}")
                    continue
            
            logger.info(f"Converted {len(documents)} documents to ProcessedDocument objects")
            
            if not documents:
                error_msg = "No documents to process"
                logger.error(error_msg)
                state.errors.append(error_msg)
                return state
            
            processed_documents = []
            total_chunks = 0
            modality_distribution = {}
            
            for document in documents:
                try:
                    processed_doc = self._process_single_document(document)
                    processed_documents.append(processed_doc)
                    total_chunks += len(processed_doc.chunks)
                    
                    # Update modality distribution
                    for chunk in processed_doc.chunks:
                        modality = chunk.modality.value
                        modality_distribution[modality] = modality_distribution.get(modality, 0) + 1
                    
                    logger.info(f"Processed document {document.document_id}: {len(processed_doc.chunks)} chunks")
                    
                except Exception as e:
                    error_msg = f"Error processing document {document.document_id}: {e}"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
            
            # Update state - convert ProcessedDocument objects to dictionaries for LangGraph serialization
            state.processed_documents = [doc.model_dump() for doc in processed_documents]
            state.total_chunks = total_chunks
            state.modality_distribution = modality_distribution
            state.processing_summary = {
                "total_documents": len(processed_documents),
                "total_chunks": total_chunks,
                "modality_distribution": modality_distribution,
                "errors": len(state.errors)
            }
            
            state.messages.append({
                "type": "info",
                "content": f"Processed {len(processed_documents)} documents, created {total_chunks} chunks"
            })
            
            logger.info(f"Document processing completed. Total chunks: {total_chunks}")
            
        except Exception as e:
            error_msg = f"Fatal error in process node: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    def end_node(self, state: EndNodeState) -> EndNodeState:
        """
        End node that finalizes the processing and provides summary.
        
        This node creates the final summary and marks processing as complete.
        """
        try:
            logger.info("Starting end node")
            
            # Get processed documents from previous node
            processed_doc_dicts = getattr(state, 'processed_documents', [])
            
            # Convert dictionaries back to ProcessedDocument objects
            processed_documents = []
            for doc_item in processed_doc_dicts:
                try:
                    if isinstance(doc_item, ProcessedDocument):
                        # Already a ProcessedDocument object
                        processed_documents.append(doc_item)
                    elif isinstance(doc_item, dict):
                        # Dictionary that needs to be converted
                        document = ProcessedDocument(**doc_item)
                        processed_documents.append(document)
                    else:
                        logger.warning(f"Unexpected processed document type: {type(doc_item)}")
                except Exception as e:
                    logger.warning(f"Error converting processed document to ProcessedDocument: {e}")
                    continue
            
            # Create final summary
            total_processed = len(processed_documents)
            total_chunks = sum(len(doc.chunks) for doc in processed_documents)
            
            # Calculate modality distribution
            modality_distribution = {}
            for doc in processed_documents:
                for chunk in doc.chunks:
                    modality = chunk.modality.value
                    modality_distribution[modality] = modality_distribution.get(modality, 0) + 1
            
            # Create summary
            summary = {
                "total_documents": total_processed,
                "total_chunks": total_chunks,
                "modality_distribution": modality_distribution,
                "processing_successful": len(state.errors) == 0,
                "error_count": len(state.errors),
                "config_used": {
                    "model_name": self.config.model_name,
                    "database_type": self.config.database_type,
                    "embedding_model": self.config.embedding_model,
                    "temperature": self.config.temperature
                }
            }
            
            # Update state
            state.final_documents = processed_documents
            state.total_processed = total_processed
            state.success = len(state.errors) == 0
            state.summary = summary
            
            state.messages.append({
                "type": "success" if state.success else "error",
                "content": f"Processing completed. {total_processed} documents, {total_chunks} chunks"
            })
            
            logger.info(f"End node completed. Success: {state.success}")
            
        except Exception as e:
            error_msg = f"Fatal error in end node: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.success = False
        
        return state
    
    def _extract_file_paths_from_config(self) -> List[Path]:
        """Extract file paths from quiz configuration."""
        file_paths = []
        
        for category in self.config.categories:
            for doc_path in category.doc_paths:
                file_path = Path(doc_path)
                if file_path.exists():
                    file_paths.append(file_path)
                else:
                    logger.warning(f"Document path not found: {doc_path}")
        
        return file_paths
    
    def _ingest_single_document(self, file_path: Path) -> ProcessedDocument:
        """Ingest a single document using the ingestion manager."""
        for ingestor in self.ingestion_manager.ingestors:
            if ingestor.can_ingest(file_path):
                return ingestor.ingest_document(file_path)
        
        raise ValueError(f"No suitable ingestor found for file: {file_path}")
    
    def _process_single_document(self, document: ProcessedDocument) -> ProcessedDocument:
        """Process a single document through the complete pipeline."""
        # Step 1: Chunk the document
        chunk_size = self.config.auto_topic_distribution_settings.vector_database_settings.get("chunk_size", 500)
        overlap_size = 200  # Default overlap
        chunks = self.chunking_strategy.chunk_document(document, chunk_size, overlap_size)
        
        # Step 2: Clean the chunks
        cleaned_chunks = self.text_cleaner.clean_chunks(chunks)
        
        # Step 3: Enrich metadata
        enriched_chunks = self.metadata_enricher.enrich_chunks(cleaned_chunks, document)
        
        # Step 4: Update document with processed chunks
        document.chunks = enriched_chunks
        document.processing_status = "processed"
        
        return document


class DocumentProcessingGraph:
    """
    LangGraph graph for document processing pipeline.
    
    Implements the graph: START → Ingest → Process → END
    """
    
    def __init__(self, config: QuizConfig):
        """Initialize the document processing graph."""
        self.config = config
        self.nodes = DocumentProcessingNodes(config)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create the graph
        workflow = StateGraph(IngestNodeState)
        
        # Add nodes
        workflow.add_node("ingest", self.nodes.ingest_documents_node)
        workflow.add_node("process", self.nodes.process_documents_node)
        workflow.add_node("end", self.nodes.end_node)
        
        # Define the flow
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "process")
        workflow.add_edge("process", "end")
        workflow.add_edge("end", END)
        
        return workflow
    
    def compile(self):
        """Compile the graph for execution."""
        return self.graph.compile()
    
    def run(self, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the document processing pipeline."""
        if initial_state is None:
            initial_state = {}
        
        # Create initial state
        state = IngestNodeState(**initial_state)
        
        # Compile and run the graph
        compiled_graph = self.compile()
        result = compiled_graph.invoke(state)
        
        return result


def create_document_processing_graph(config_path: str = "config/quiz_config.json") -> DocumentProcessingGraph:
    """
    Factory function to create document processing graph from config file.
    
    Args:
        config_path: Path to the quiz configuration file
        
    Returns:
        DocumentProcessingGraph instance
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = QuizConfig(**config_data)
        
        # Create and return graph
        return DocumentProcessingGraph(config)
        
    except Exception as e:
        logger.error(f"Error creating document processing graph: {e}")
        raise 