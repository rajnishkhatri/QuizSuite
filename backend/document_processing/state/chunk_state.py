"""
Chunk Node State

State for chunking node (Step 3).
"""

from typing import List, Dict, Any, Annotated, Optional
from pydantic import Field
from .base_state import BaseState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class ChunkNodeState(BaseState):
    """
    State model for chunking node (Step 3).
    
    This state contains:
    - Cleaned documents to chunk
    - Extracted content results
    - Created chunks
    - Chunking statistics
    """
    
    # Cleaned documents to chunk
    cleaned_documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Cleaned documents to chunk")
    
    # Extracted content results (from Step 1)
    extracted_content_results: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Extracted content results from Step 1")
    
    # Created chunks
    chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Created chunks")
    
    # Chunking statistics
    total_chunks: int = Field(default=0, description="Total number of chunks created")
    
    # Chunking results
    chunking_results: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Results of chunking process")
    
    # Processing metadata
    chunking_summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Summary of chunking process")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True 