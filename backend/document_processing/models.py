"""
Pydantic models for document processing state management.

This module defines the data structures used throughout the document processing pipeline.
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentType(str, Enum):
    """Supported document types for processing."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"


class ModalityType(str, Enum):
    """Types of content modalities in documents."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"


class DocumentChunk(BaseModel):
    """Represents a processed chunk of document content."""
    content: str = Field(..., description="The text content of the chunk")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="Source document identifier")
    modality: ModalityType = Field(..., description="Type of content modality")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    chunk_index: int = Field(..., description="Order of chunk in document")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")


class ProcessedDocument(BaseModel):
    """Represents a fully processed document."""
    document_id: str = Field(..., description="Unique document identifier")
    file_path: Path = Field(..., description="Path to the source file")
    document_type: DocumentType = Field(..., description="Type of the document")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Processed chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    processing_status: str = Field(default="pending", description="Processing status")


class ChromaDBSettings(BaseModel):
    """ChromaDB configuration settings."""
    pdf_persist_directory: str = Field(..., description="PDF database persistence directory")
    html_persist_directory: str = Field(..., description="HTML database persistence directory")
    pdf_collection_name: str = Field(..., description="PDF collection name")
    html_collection_name: str = Field(..., description="HTML collection name")
    use_existing: bool = Field(default=True, description="Use existing database")
    create_if_not_exists: bool = Field(default=True, description="Create database if not exists")
    load_existing: bool = Field(default=True, description="Load existing data")
    use_metadata_filtering: bool = Field(default=True, description="Use metadata filtering")
    max_retrieval_results: int = Field(default=20, description="Maximum retrieval results")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity")


class CategoryConfig(BaseModel):
    """Configuration for a quiz category."""
    name: str = Field(..., description="Category name")
    num_questions: int = Field(..., description="Number of questions to generate")
    description: str = Field(..., description="Category description")
    doc_paths: List[str] = Field(..., description="Document paths for this category")


class AutoTopicDistributionSettings(BaseModel):
    """Settings for automatic topic distribution."""
    enabled: bool = Field(default=True, description="Enable auto topic distribution")
    vector_database_settings: Dict[str, Any] = Field(default_factory=dict)
    coverage_settings: Dict[str, Any] = Field(default_factory=dict)
    question_generation_settings: Dict[str, Any] = Field(default_factory=dict)
    performance_settings: Dict[str, Any] = Field(default_factory=dict)


class CacheSettings(BaseModel):
    """Cache configuration settings."""
    enabled: bool = Field(default=True, description="Enable caching")
    cache_dir: str = Field(..., description="Cache directory")
    compression_ratio: int = Field(default=50, description="Compression ratio")
    retention_target: int = Field(default=95, description="Retention target percentage")


class OutputSettings(BaseModel):
    """Output configuration settings."""
    output_dir: str = Field(..., description="Output directory")
    include_timestamp: bool = Field(default=True, description="Include timestamp in output")
    include_model_name: bool = Field(default=True, description="Include model name in output")
    include_database_type: bool = Field(default=True, description="Include database type in output")


class PromptSettings(BaseModel):
    """Prompt configuration settings."""
    use_advanced_prompt: bool = Field(default=True, description="Use advanced prompts")
    include_learning_objectives: bool = Field(default=True, description="Include learning objectives")
    include_target_audience: bool = Field(default=True, description="Include target audience")
    include_difficulty_level: bool = Field(default=True, description="Include difficulty level")


class QuizConfig(BaseModel):
    """Main quiz configuration model."""
    database_type: str = Field(..., description="Database type")
    model_name: str = Field(..., description="Model name")
    embedding_model: str = Field(..., description="Embedding model")
    temperature: float = Field(default=0.0, description="Model temperature")
    target_audience: str = Field(..., description="Target audience")
    difficulty_level: str = Field(..., description="Difficulty level")
    learning_objectives: str = Field(..., description="Learning objectives")
    max_context_tokens: int = Field(default=1000, description="Maximum context tokens")
    use_memory: bool = Field(default=False, description="Use memory")
    use_reflexion: bool = Field(default=False, description="Use reflexion")
    chroma_db_settings: ChromaDBSettings = Field(..., description="ChromaDB settings")
    categories: List[CategoryConfig] = Field(..., description="Quiz categories")
    auto_topic_distribution_settings: AutoTopicDistributionSettings = Field(..., description="Auto topic distribution settings")
    cache_settings: CacheSettings = Field(..., description="Cache settings")
    output_settings: OutputSettings = Field(..., description="Output settings")
    prompt_settings: PromptSettings = Field(..., description="Prompt settings")


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


class NodeState(BaseModel):
    """Base state for LangGraph nodes."""
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Node messages")
    errors: List[str] = Field(default_factory=list, description="Node errors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Node timestamp")


class IngestNodeState(NodeState):
    """State for document ingestion node."""
    documents: List[ProcessedDocument] = Field(default_factory=list, description="Ingested documents")
    file_paths: List[Path] = Field(default_factory=list, description="File paths to process")
    config: Optional[QuizConfig] = Field(None, description="Quiz configuration")


class ProcessNodeState(NodeState):
    """State for document processing node."""
    processed_documents: List[ProcessedDocument] = Field(default_factory=list, description="Processed documents")
    total_chunks: int = Field(default=0, description="Total chunks processed")
    modality_distribution: Dict[str, int] = Field(default_factory=dict, description="Modality distribution")
    processing_summary: Dict[str, Any] = Field(default_factory=dict, description="Processing summary")


class EndNodeState(NodeState):
    """State for end node."""
    final_documents: List[ProcessedDocument] = Field(default_factory=list, description="Final processed documents")
    total_processed: int = Field(default=0, description="Total documents processed")
    success: bool = Field(default=False, description="Whether processing was successful")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Final summary") 