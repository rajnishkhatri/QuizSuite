"""
LangSmith Tracing Utilities

Provides tracing and monitoring capabilities for the document processing pipeline.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from langsmith import Client, RunTree, traceable
from langsmith.run_helpers import trace

logger = logging.getLogger(__name__)


class DocumentProcessingTracer:
    """Tracer for document processing pipeline."""
    
    def __init__(self, project_name: str = "quiz-suite-document-processing"):
        """Initialize the tracer.
        
        Args:
            project_name: Name of the LangSmith project
        """
        self.project_name = project_name
        self.client = Client()
        
        # Set up environment variables for LangSmith
        if not os.getenv("LANGCHAIN_TRACING_V2"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        if not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = project_name
    
    def trace_ingestion(self, file_paths: List[Path], config: Dict[str, Any]) -> RunTree:
        """Trace document ingestion phase.
        
        Args:
            file_paths: List of file paths to ingest
            config: Configuration used for ingestion
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="document_ingestion",
            project_name=self.project_name,
            tags=["ingestion", "document_processing"]
        ) as tracer:
            tracer.inputs = {
                "file_paths": [str(p) for p in file_paths],
                "file_count": len(file_paths),
                "config": config
            }
            
            logger.info(f"Starting ingestion trace for {len(file_paths)} files")
            return tracer
    
    def trace_processing(self, documents: List[Dict[str, Any]], config: Dict[str, Any]) -> RunTree:
        """Trace document processing phase.
        
        Args:
            documents: List of documents to process
            config: Configuration used for processing
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="document_processing",
            project_name=self.project_name,
            tags=["processing", "document_processing"]
        ) as tracer:
            tracer.inputs = {
                "document_count": len(documents),
                "document_ids": [doc.get("document_id", "unknown") for doc in documents],
                "config": config
            }
            
            logger.info(f"Starting processing trace for {len(documents)} documents")
            return tracer
    
    def trace_chunking(self, document_id: str, chunk_count: int, modality_distribution: Dict[str, int]) -> RunTree:
        """Trace document chunking phase.
        
        Args:
            document_id: ID of the document being chunked
            chunk_count: Number of chunks created
            modality_distribution: Distribution of content modalities
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="document_chunking",
            project_name=self.project_name,
            tags=["chunking", "document_processing"]
        ) as tracer:
            tracer.inputs = {
                "document_id": document_id,
                "chunk_count": chunk_count,
                "modality_distribution": modality_distribution
            }
            
            logger.info(f"Starting chunking trace for document {document_id}")
            return tracer
    
    def trace_cleaning(self, document_id: str, chunk_count: int, cleaning_stats: Dict[str, Any]) -> RunTree:
        """Trace text cleaning phase.
        
        Args:
            document_id: ID of the document being cleaned
            chunk_count: Number of chunks cleaned
            cleaning_stats: Statistics about the cleaning process
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="text_cleaning",
            project_name=self.project_name,
            tags=["cleaning", "document_processing"]
        ) as tracer:
            tracer.inputs = {
                "document_id": document_id,
                "chunk_count": chunk_count,
                "cleaning_stats": cleaning_stats
            }
            
            logger.info(f"Starting cleaning trace for document {document_id}")
            return tracer
    
    def trace_enrichment(self, document_id: str, chunk_count: int, enrichment_stats: Dict[str, Any]) -> RunTree:
        """Trace metadata enrichment phase.
        
        Args:
            document_id: ID of the document being enriched
            chunk_count: Number of chunks enriched
            enrichment_stats: Statistics about the enrichment process
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="metadata_enrichment",
            project_name=self.project_name,
            tags=["enrichment", "document_processing"]
        ) as tracer:
            tracer.inputs = {
                "document_id": document_id,
                "chunk_count": chunk_count,
                "enrichment_stats": enrichment_stats
            }
            
            logger.info(f"Starting enrichment trace for document {document_id}")
            return tracer
    
    def trace_embedding(self, embedding_config: Dict[str, Any]) -> RunTree:
        """Trace embedding generation and storage phase.
        
        Args:
            embedding_config: Configuration for embedding generation
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="embedding_generation",
            project_name=self.project_name,
            tags=["embedding", "document_processing"]
        ) as tracer:
            tracer.inputs = embedding_config
            
            logger.info("Starting embedding generation trace")
            return tracer
    
    def trace_error(self, error: Exception, context: Dict[str, Any]) -> RunTree:
        """Trace errors in the pipeline.
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="document_processing_error",
            project_name=self.project_name,
            tags=["error", "document_processing"]
        ) as tracer:
            tracer.inputs = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context
            }
            
            logger.error(f"Tracing error: {error} in context: {context}")
            return tracer
    
    def trace_completion(self, summary: Dict[str, Any]) -> RunTree:
        """Trace pipeline completion.
        
        Args:
            summary: Summary of the processing results
            
        Returns:
            RunTree for tracing
        """
        with trace(
            name="document_processing_completion",
            project_name=self.project_name,
            tags=["completion", "document_processing"]
        ) as tracer:
            tracer.inputs = summary
            
            logger.info(f"Tracing completion: {summary}")
            return tracer


@traceable(name="document_processing_pipeline")
def trace_document_processing_pipeline(
    file_paths: List[Path],
    config: Dict[str, Any],
    processing_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Trace the complete document processing pipeline.
    
    Args:
        file_paths: List of input file paths
        config: Configuration used for processing
        processing_results: Results of the processing pipeline
        
    Returns:
        Tracing results
    """
    return {
        "input_file_count": len(file_paths),
        "input_files": [str(p) for p in file_paths],
        "config_summary": {
            "model_name": config.get("model_name"),
            "database_type": config.get("database_type"),
            "temperature": config.get("temperature")
        },
        "processing_results": processing_results,
        "timestamp": datetime.utcnow().isoformat()
    } 