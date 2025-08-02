"""
Unified State Module

This module defines a unified state model that uses composition to pass only necessary data
between nodes, avoiding inheritance issues.
"""

from typing import List, Dict, Any, Optional, Annotated
from pathlib import Path
from pydantic import Field

from .base_state import BaseNodeState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class UnifiedPipelineState(BaseNodeState):
    """
    Unified state for the entire pipeline using composition.
    
    This state contains only the essential data that needs to be passed between nodes:
    - Configuration (shared across all nodes)
    - Documents (from Ingest to Process)
    - Chunks (from Process to Embed)
    - Embedded chunks (from Embed to Storage)
    - Final results (for end node)
    
    All other data is kept private within each node's state.
    """
    
    # Configuration (shared across all nodes)
    config: Optional[Dict[str, Any]] = Field(None, description="Pipeline configuration")
    
    # Data passed from Ingest to Process
    documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Documents from ingest")
    total_documents: int = Field(default=0, description="Total documents count")
    
    # Data passed from Process to Embed
    chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Chunks from process")
    total_chunks: int = Field(default=0, description="Total chunks count")
    
    # Data passed from Embed to Storage
    embedded_chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Embedded chunks from embed")
    total_embedded: int = Field(default=0, description="Total embedded chunks count")
    
    # Data passed from Storage to End
    stored_chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Stored chunks from storage")
    total_stored: int = Field(default=0, description="Total stored chunks count")
    
    # Final pipeline results
    pipeline_success: bool = Field(default=False, description="Overall pipeline success")
    final_summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Final pipeline summary")
    
    # Helper methods for data transfer
    def set_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Set documents from ingest node."""
        self.documents = documents
        self.total_documents = len(documents)
    
    def set_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Set chunks from process node."""
        self.chunks = chunks
        self.total_chunks = len(chunks)
    
    def set_embedded_chunks(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        """Set embedded chunks from embed node."""
        self.embedded_chunks = embedded_chunks
        self.total_embedded = len(embedded_chunks)
    
    def set_stored_chunks(self, stored_chunks: List[Dict[str, Any]]) -> None:
        """Set stored chunks from storage node."""
        self.stored_chunks = stored_chunks
        self.total_stored = len(stored_chunks)
    
    def set_final_summary(self, summary: Dict[str, Any]) -> None:
        """Set final pipeline summary."""
        self.final_summary = summary
        self.pipeline_success = True 