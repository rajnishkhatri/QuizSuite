"""
State Models

Pydantic models for document processing state management.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field

from .document_models import ProcessedDocument
from .config_models import QuizConfig


class DocumentProcessingState(BaseModel):
    """State model for document processing pipeline."""
    documents: List[ProcessedDocument] = Field(default_factory=list, description="Processed documents")
    total_chunks: int = Field(default=0, description="Total number of chunks processed")
    processing_errors: List[str] = Field(default_factory=list, description="Processing errors")
    is_complete: bool = Field(default=False, description="Whether processing is complete")
    config: Optional[QuizConfig] = Field(None, description="Quiz configuration")
    current_category: Optional[str] = Field(None, description="Current category being processed")
    processed_categories: List[str] = Field(default_factory=list, description="Categories that have been processed")
    embeddings_generated: bool = Field(default=False, description="Whether embeddings have been generated")
    vector_db_initialized: bool = Field(default=False, description="Whether vector database is initialized")


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    file_paths: List[Path] = Field(..., description="Paths to documents to ingest")
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    overlap_size: int = Field(default=200, description="Overlap between chunks")
    include_images: bool = Field(default=True, description="Whether to process images")
    include_tables: bool = Field(default=True, description="Whether to process tables")
    config: Optional[QuizConfig] = Field(None, description="Quiz configuration") 