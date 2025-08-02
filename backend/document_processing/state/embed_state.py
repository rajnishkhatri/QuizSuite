"""
Embed State Module

Independent state for embedding generation node with private fields and minimal output.
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import Field
from .base_state import BaseNodeState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class EmbedNodeState(BaseNodeState):
    """
    Independent state for embedding generation node.
    
    Private fields for internal processing:
    - embedding_stats: Internal embedding statistics
    - embedding_model: Internal model information
    - embedding_device: Internal device information
    - embedding_error: Internal error information
    
    Output fields passed to next node:
    - embedded_chunks: List of embedded chunks for next node
    - total_embedded: Count for next node
    """
    
    # Private fields (internal processing)
    embedding_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Internal embedding statistics")
    embedding_model: Optional[str] = Field(None, description="Internal embedding model")
    embedding_dimension: Optional[int] = Field(None, description="Internal embedding dimension")
    embedding_device: Optional[str] = Field(None, description="Internal embedding device")
    embedding_error: Optional[str] = Field(None, description="Internal embedding error")
    processing_status: str = Field(default="pending", description="Internal processing status")
    
    # Output fields (passed to next node)
    embedded_chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Embedded chunks for next node")
    total_embedded: int = Field(default=0, description="Total embedded chunks count for next node")
    
    def add_embedded_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add an embedded chunk to the output list."""
        self.embedded_chunks.append(chunk)
        self.total_embedded = len(self.embedded_chunks)
    
    def set_embedding_stats(self, stats: Dict[str, Any]) -> None:
        """Set internal embedding statistics."""
        self.embedding_stats = stats
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get internal embedding statistics."""
        return self.embedding_stats
    
    def set_embedding_model(self, model: str) -> None:
        """Set internal embedding model."""
        self.embedding_model = model
    
    def get_embedding_model(self) -> Optional[str]:
        """Get internal embedding model."""
        return self.embedding_model
    
    def set_embedding_dimension(self, dimension: int) -> None:
        """Set internal embedding dimension."""
        self.embedding_dimension = dimension
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get internal embedding dimension."""
        return self.embedding_dimension
    
    def set_embedding_device(self, device: str) -> None:
        """Set internal embedding device."""
        self.embedding_device = device
    
    def get_embedding_device(self) -> Optional[str]:
        """Get internal embedding device."""
        return self.embedding_device
    
    def set_embedding_error(self, error: str) -> None:
        """Set internal embedding error."""
        self.embedding_error = error
    
    def get_embedding_error(self) -> Optional[str]:
        """Get internal embedding error."""
        return self.embedding_error
    
    def set_processing_status(self, status: str) -> None:
        """Set internal processing status."""
        self.processing_status = status
    
    def get_processing_status(self) -> str:
        """Get internal processing status."""
        return self.processing_status 