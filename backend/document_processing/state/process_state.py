"""
Process Node State

Independent state for document processing node with private fields and minimal output.
"""

from typing import List, Dict, Any, Optional, Annotated
from pathlib import Path
from pydantic import Field

from .base_state import BaseNodeState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class ProcessNodeState(BaseNodeState):
    """
    Independent state for document processing node.
    
    Private fields for internal processing:
    - processed_documents: Internal processed document list
    - extracted_content: Internal content extraction results
    - processing_summary: Internal processing statistics
    
    Output fields passed to next node:
    - chunks: List of document chunks for next node
    - total_chunks: Count for next node
    """
    
    # Private fields (internal processing)
    processed_documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Internal processed documents")
    extracted_content: Optional[Dict[str, Any]] = Field(None, description="Internal extracted content")
    processing_summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Internal processing summary")
    processing_status: str = Field(default="pending", description="Internal processing status")
    
    # Output fields (passed to next node)
    chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Document chunks for next node")
    total_chunks: int = Field(default=0, description="Total chunks count for next node")
    
    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a chunk to the output list."""
        self.chunks.append(chunk)
        self.total_chunks = len(self.chunks)
    
    def add_processed_document(self, document: Dict[str, Any]) -> None:
        """Add a processed document to internal list."""
        self.processed_documents.append(document)
    
    def get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get internal processed documents."""
        return self.processed_documents
    
    def set_extracted_content(self, content: Dict[str, Any]) -> None:
        """Set internal extracted content."""
        self.extracted_content = content
    
    def get_extracted_content(self) -> Optional[Dict[str, Any]]:
        """Get internal extracted content."""
        return self.extracted_content
    
    def set_processing_summary(self, summary: Dict[str, Any]) -> None:
        """Set internal processing summary."""
        self.processing_summary = summary
    
    def get_processing_summary(self) -> Optional[Dict[str, Any]]:
        """Get internal processing summary."""
        return self.processing_summary
    
    def set_processing_status(self, status: str) -> None:
        """Set internal processing status."""
        self.processing_status = status
    
    def get_processing_status(self) -> str:
        """Get internal processing status."""
        return self.processing_status 