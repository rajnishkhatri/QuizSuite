"""
Storage State Module

Independent state for storage node with private fields and minimal output.
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import Field
from .base_state import BaseNodeState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class StorageNodeState(BaseNodeState):
    """
    Independent state for storage node.
    
    Private fields for internal processing:
    - storage_info: Internal storage information
    - collection_stats: Internal collection statistics
    - storage_error: Internal storage error
    - search_results: Internal search results
    
    Output fields passed to next node:
    - stored_chunks: List of stored chunks for next node
    - total_stored: Count for next node
    """
    
    # Private fields (internal processing)
    storage_info: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Internal storage information")
    collection_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Internal collection statistics")
    storage_error: Optional[str] = Field(None, description="Internal storage error")
    search_results: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Internal search results")
    processing_status: str = Field(default="pending", description="Internal processing status")
    
    # Output fields (passed to next node)
    stored_chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Stored chunks for next node")
    total_stored: int = Field(default=0, description="Total stored chunks count for next node")
    
    def add_stored_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a stored chunk to the output list."""
        self.stored_chunks.append(chunk)
        self.total_stored = len(self.stored_chunks)
    
    def set_storage_info(self, info: Dict[str, Any]) -> None:
        """Set internal storage information."""
        self.storage_info = info
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get internal storage information."""
        return self.storage_info
    
    def set_collection_stats(self, stats: Dict[str, Any]) -> None:
        """Set internal collection statistics."""
        self.collection_stats = stats
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get internal collection statistics."""
        return self.collection_stats
    
    def set_storage_error(self, error: str) -> None:
        """Set internal storage error."""
        self.storage_error = error
    
    def get_storage_error(self) -> Optional[str]:
        """Get internal storage error."""
        return self.storage_error
    
    def add_search_result(self, result: Dict[str, Any]) -> None:
        """Add a search result to internal list."""
        self.search_results.append(result)
    
    def get_search_results(self) -> List[Dict[str, Any]]:
        """Get internal search results."""
        return self.search_results
    
    def set_processing_status(self, status: str) -> None:
        """Set internal processing status."""
        self.processing_status = status
    
    def get_processing_status(self) -> str:
        """Get internal processing status."""
        return self.processing_status 