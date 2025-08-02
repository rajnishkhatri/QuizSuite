"""
Ingest Node State

Independent state for document ingestion node with private fields and minimal output.
"""

from typing import List, Optional, Annotated, Dict, Any
from pathlib import Path
from pydantic import Field

from .base_state import BaseNodeState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class IngestNodeState(BaseNodeState):
    """
    Independent state for document ingestion node.
    
    Private fields for internal processing:
    - file_paths: Internal list of files to process
    - document_metadata: Internal document information
    
    Output fields passed to next node:
    - documents: List of document dictionaries for next node
    - total_documents: Count for next node
    """
    
    # Private fields (internal processing)
    file_paths: Annotated[List[Path], list_merge_reducer] = Field(default_factory=list, description="Internal file paths")
    document_metadata: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Internal document metadata")
    processing_status: str = Field(default="pending", description="Internal processing status")
    
    # Output fields (passed to next node)
    documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Documents for next node")
    total_documents: int = Field(default=0, description="Total documents count for next node")
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """Add a document to the output list."""
        self.documents.append(document)
        self.total_documents = len(self.documents)
    
    def set_file_paths(self, paths: List[Path]) -> None:
        """Set internal file paths."""
        self.file_paths = paths
    
    def get_file_paths(self) -> List[Path]:
        """Get internal file paths."""
        return self.file_paths
    
    def set_processing_status(self, status: str) -> None:
        """Set internal processing status."""
        self.processing_status = status
    
    def get_processing_status(self) -> str:
        """Get internal processing status."""
        return self.processing_status 