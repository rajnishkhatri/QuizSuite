"""
Document Processing Nodes

LangGraph nodes for document processing pipeline.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from ..state.ingest_state import IngestNodeState
from ..state.process_state import ProcessNodeState
from ..state.embed_state import EmbedNodeState
from ..state.end_state import EndNodeState
from ..model.config_models import QuizConfig
from ..model.document_models import ProcessedDocument, DocumentType
from ..processor.document_ingestor import DocumentIngestionManager
from ..processor.chunking_strategy import ModalityAwareChunkingStrategy
from ..processor.metadata_enricher import MetadataEnricher
from ..processor.text_cleaner import ModalityAwareTextCleaner
from ..processor.embedding_manager import EmbeddingManager
from ..processor.storage_manager import StorageManager
from ..utils.tracing import DocumentProcessingTracer
from ..utils.error_handling import (
    ErrorHandler,
    error_handler,
    safe_execute,
    validate_file_path,
    validate_config,
    IngestionError,
    ProcessingError,
    ChunkingError,
    CleaningError,
    EnrichmentError
)


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
        self.embedding_manager = EmbeddingManager(
            embedding_model=self.config.embedding_model
        )
        self.storage_manager = StorageManager(self.config.model_dump())
        
        # Initialize tracing and error handling
        self.tracer = DocumentProcessingTracer()
        self.error_handler = ErrorHandler(self.tracer)
        
        # Validate configuration
        try:
            validate_config(self.config.dict())
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def ingest_documents_node(self, state: IngestNodeState) -> IngestNodeState:
        """
        Node for ingesting documents.
        
        This node handles the ingestion of documents from various sources.
        """
        # Start tracing
        with self.tracer.trace_ingestion([], self.config.dict()) as tracer:
            try:
                logger.info("Starting document ingestion node")
                
                # Extract file paths from config categories
                file_paths = self._extract_file_paths_from_config()
                
                if not file_paths:
                    error_msg = "No valid file paths found in configuration"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
                    return state
                
                # Update tracer with actual file paths
                tracer.inputs["file_paths"] = [str(p) for p in file_paths]
                tracer.inputs["file_count"] = len(file_paths)
                
                # Ingest documents with error handling
                documents = []
                failed_files = []
                
                for file_path in file_paths:
                    try:
                        # Validate file path
                        validate_file_path(file_path)
                        
                        # Ingest document with retry logic
                        document = safe_execute(
                            self._ingest_single_document,
                            file_path,
                            error_type="ingestion",
                            context={"file_path": str(file_path)},
                            default_return=None
                        )
                        
                        if document:
                            documents.append(document)
                            logger.info(f"Successfully ingested: {file_path}")
                        else:
                            failed_files.append(str(file_path))
                            
                    except Exception as e:
                        error_msg = f"Error ingesting {file_path}: {e}"
                        logger.error(error_msg)
                        state.errors.append(error_msg)
                        failed_files.append(str(file_path))
                        
                        # Trace the error
                        self.error_handler.handle_error(
                            e, 
                            {"file_path": str(file_path), "node": "ingest_documents_node"},
                            "ingestion",
                            reraise=False
                        )
                
                # Update state
                state.documents = documents
                state.file_paths = file_paths
                state.config = self.config
                
                # Add success/failure messages
                if documents:
                    state.messages.append({
                        "type": "success",
                        "content": f"Ingested {len(documents)} documents from {len(file_paths)} file paths"
                    })
                
                if failed_files:
                    state.messages.append({
                        "type": "warning",
                        "content": f"Failed to ingest {len(failed_files)} files: {failed_files}"
                    })
                
                # Update tracer outputs
                tracer.outputs = {
                    "successful_ingestions": len(documents),
                    "failed_ingestions": len(failed_files),
                    "total_files": len(file_paths),
                    "failed_files": failed_files
                }
                
                logger.info(f"Document ingestion completed. Processed {len(documents)} documents")
                
            except Exception as e:
                error_msg = f"Fatal error in ingest node: {e}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                
                # Trace the fatal error
                self.error_handler.handle_error(
                    e,
                    {"node": "ingest_documents_node", "state": "fatal"},
                    "ingestion",
                    reraise=False
                )
        
        return state
    
    def process_documents_node(self, state: ProcessNodeState) -> ProcessNodeState:
        """
        Node for processing documents (chunking, cleaning, metadata enrichment).
        
        This node handles the complete processing pipeline for each document.
        """
        # Start tracing
        with self.tracer.trace_processing([], self.config.dict()) as tracer:
            try:
                logger.info("Starting document processing node")
                
                # Get documents from previous node
                documents = state.get("documents", [])
                
                if not documents:
                    error_msg = "No documents to process"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
                    return state
                
                # Update tracer with document information
                tracer.inputs["document_count"] = len(documents)
                tracer.inputs["document_ids"] = [doc.document_id for doc in documents]
                
                processed_documents = []
                total_chunks = 0
                modality_distribution = {}
                failed_documents = []
                
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
                        failed_documents.append(document.document_id)
                        
                        # Trace the error
                        self.error_handler.handle_error(
                            e,
                            {"document_id": document.document_id, "node": "process_documents_node"},
                            "processing",
                            reraise=False
                        )
                
                # Update state
                state.processed_documents = processed_documents
                state.total_chunks = total_chunks
                state.modality_distribution = modality_distribution
                state.processing_summary = {
                    "total_documents": len(processed_documents),
                    "total_chunks": total_chunks,
                    "modality_distribution": modality_distribution,
                    "failed_documents": len(failed_documents),
                    "errors": len(state.errors)
                }
                
                # Add success/failure messages
                if processed_documents:
                    state.messages.append({
                        "type": "success",
                        "content": f"Processed {len(processed_documents)} documents, created {total_chunks} chunks"
                    })
                
                if failed_documents:
                    state.messages.append({
                        "type": "warning",
                        "content": f"Failed to process {len(failed_documents)} documents: {failed_documents}"
                    })
                
                # Update tracer outputs
                tracer.outputs = {
                    "successful_processing": len(processed_documents),
                    "failed_processing": len(failed_documents),
                    "total_chunks": total_chunks,
                    "modality_distribution": modality_distribution,
                    "failed_documents": failed_documents
                }
                
                logger.info(f"Document processing completed. Total chunks: {total_chunks}")
                
            except Exception as e:
                error_msg = f"Fatal error in process node: {e}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                
                # Trace the fatal error
                self.error_handler.handle_error(
                    e,
                    {"node": "process_documents_node", "state": "fatal"},
                    "processing",
                    reraise=False
                )
        
        return state
    
    def embed_and_store_node(self, state: EmbedNodeState) -> EmbedNodeState:
        """
        Node for generating embeddings and storing in databases.
        
        This node handles embedding generation and storage in vector/graph databases.
        """
        # Start tracing
        with self.tracer.trace_embedding({}) as tracer:
            try:
                logger.info("Starting embed and store node")
                
                # Get processed documents from previous node
                processed_documents = state.get("processed_documents", [])
                
                if not processed_documents:
                    error_msg = "No processed documents available for embedding"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
                    return state
                
                # Extract all chunks from processed documents
                all_chunks = []
                for doc in processed_documents:
                    all_chunks.extend(doc.chunks)
                
                logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
                
                # Generate embeddings
                embedded_chunks = self.embedding_manager.generate_embeddings(all_chunks)
                
                # Update documents with embedded chunks
                embedded_documents = []
                chunk_index = 0
                
                for doc in processed_documents:
                    doc_chunks = embedded_chunks[chunk_index:chunk_index + len(doc.chunks)]
                    embedded_doc = doc.model_copy(update={"chunks": doc_chunks})
                    embedded_documents.append(embedded_doc)
                    chunk_index += len(doc.chunks)
                
                # Store in vector database
                vector_result = self.storage_manager.store_in_vector_database(embedded_chunks)
                
                # Store in graph database
                graph_result = self.storage_manager.store_in_graph_database(embedded_documents)
                
                # Get embedding statistics
                embedding_stats = self.embedding_manager.get_embedding_stats(embedded_chunks)
                
                # Update state
                state.processed_documents = embedded_documents
                state.embedded_chunks = embedded_chunks
                state.embedding_stats = embedding_stats
                state.storage_metadata = {
                    "vector_database": vector_result,
                    "graph_database": graph_result
                }
                state.vector_database_status = vector_result.get("status", "unknown")
                state.graph_database_status = graph_result.get("status", "unknown")
                state.total_embeddings = len(embedded_chunks)
                state.embedding_model = self.config.embedding_model
                
                # Add success message
                state.messages.append({
                    "type": "success",
                    "content": f"Embedding and storage completed. {len(embedded_chunks)} chunks embedded and stored"
                })
                
                # Update tracer outputs
                tracer.outputs = {
                    "embedded_chunks": len(embedded_chunks),
                    "embedding_stats": embedding_stats,
                    "storage_results": {
                        "vector": vector_result,
                        "graph": graph_result
                    }
                }
                
                logger.info(f"Embed and store node completed. {len(embedded_chunks)} chunks processed")
                
            except Exception as e:
                error_msg = f"Fatal error in embed and store node: {e}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                
                # Trace the fatal error
                self.error_handler.handle_error(
                    e,
                    {"node": "embed_and_store_node", "state": "fatal"},
                    "embedding",
                    reraise=False
                )
        
        return state
    
    def end_node(self, state: EndNodeState) -> EndNodeState:
        """
        End node that finalizes the processing and provides summary.
        
        This node creates the final summary and marks processing as complete.
        """
        # Start tracing
        with self.tracer.trace_completion({}) as tracer:
            try:
                logger.info("Starting end node")
                
                # Get processed documents from previous node
                processed_documents = state.get("processed_documents", [])
                
                # Create final summary
                total_processed = len(processed_documents)
                total_chunks = sum(len(doc.chunks) for doc in processed_documents)
                
                # Calculate modality distribution
                modality_distribution = {}
                for doc in processed_documents:
                    for chunk in doc.chunks:
                        modality = chunk.modality.value
                        modality_distribution[modality] = modality_distribution.get(modality, 0) + 1
                
                # Get error summary
                error_summary = self.error_handler.get_error_summary()
                
                # Create summary
                summary = {
                    "total_documents": total_processed,
                    "total_chunks": total_chunks,
                    "modality_distribution": modality_distribution,
                    "processing_successful": len(state.errors) == 0,
                    "error_count": len(state.errors),
                    "error_summary": error_summary,
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
                
                # Add completion message
                if state.success:
                    state.messages.append({
                        "type": "success",
                        "content": f"Processing completed successfully. {total_processed} documents, {total_chunks} chunks"
                    })
                else:
                    state.messages.append({
                        "type": "error",
                        "content": f"Processing completed with errors. {total_processed} documents, {total_chunks} chunks, {len(state.errors)} errors"
                    })
                
                # Update tracer outputs
                tracer.outputs = summary
                
                logger.info(f"End node completed. Success: {state.success}")
                
            except Exception as e:
                error_msg = f"Fatal error in end node: {e}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                state.success = False
                
                # Trace the fatal error
                self.error_handler.handle_error(
                    e,
                    {"node": "end_node", "state": "fatal"},
                    "other",
                    reraise=False
                )
        
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
        
        raise IngestionError(f"No suitable ingestor found for file: {file_path}")
    
    def _process_single_document(self, document: ProcessedDocument) -> ProcessedDocument:
        """Process a single document through the complete pipeline."""
        try:
            # Step 1: Chunk the document with tracing
            with self.tracer.trace_chunking(
                document.document_id, 
                0, 
                {}
            ) as chunk_tracer:
                chunks = safe_execute(
                    self.chunking_strategy.chunk_document,
                    document,
                    error_type="chunking",
                    context={"document_id": document.document_id},
                    default_return=[]
                )
                
                if not chunks:
                    raise ChunkingError(f"No chunks created for document {document.document_id}")
                
                chunk_tracer.inputs["chunk_count"] = len(chunks)
                chunk_tracer.inputs["modality_distribution"] = {
                    chunk.modality.value: sum(1 for c in chunks if c.modality == chunk.modality)
                    for chunk in chunks
                }
            
            # Step 2: Clean the chunks with tracing
            with self.tracer.trace_cleaning(
                document.document_id,
                len(chunks),
                {}
            ) as clean_tracer:
                cleaned_chunks = safe_execute(
                    self.text_cleaner.clean_chunks,
                    chunks,
                    error_type="cleaning",
                    context={"document_id": document.document_id, "chunk_count": len(chunks)},
                    default_return=chunks
                )
                
                clean_tracer.inputs["cleaning_stats"] = {
                    "original_chunks": len(chunks),
                    "cleaned_chunks": len(cleaned_chunks),
                    "cleaning_strategy": "modality_aware"
                }
            
            # Step 3: Enrich metadata with tracing
            with self.tracer.trace_enrichment(
                document.document_id,
                len(cleaned_chunks),
                {}
            ) as enrich_tracer:
                enriched_chunks = safe_execute(
                    self.metadata_enricher.enrich_chunks,
                    cleaned_chunks,
                    error_type="enrichment",
                    context={"document_id": document.document_id, "chunk_count": len(cleaned_chunks)},
                    default_return=cleaned_chunks
                )
                
                enrich_tracer.inputs["enrichment_stats"] = {
                    "input_chunks": len(cleaned_chunks),
                    "enriched_chunks": len(enriched_chunks),
                    "enrichment_strategy": "comprehensive"
                }
            
            # Step 4: Update document with processed chunks
            document.chunks = enriched_chunks
            document.processing_status = "processed"
            
            return document
            
        except Exception as e:
            # Re-raise with proper error type
            if isinstance(e, (ChunkingError, CleaningError, EnrichmentError)):
                raise
            else:
                raise ProcessingError(f"Error processing document {document.document_id}: {e}") 