"""
Document Models

Pydantic models for document and chunk representation.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field


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