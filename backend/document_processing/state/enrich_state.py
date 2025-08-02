"""
Enrich Node State

State for chunk enrichment node (Step 4).
"""

from typing import List, Dict, Any, Annotated, Optional
from pydantic import Field
from .base_state import BaseState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class EnrichNodeState(BaseState):
    """
    State model for chunk enrichment node (Step 4).
    
    This state contains:
    - Chunks to enrich
    - Extracted content results
    - Enriched chunks
    - Enrichment statistics
    """
    
    # Chunks to enrich
    chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Chunks to enrich with metadata")
    
    # Extracted content results (from Step 1)
    extracted_content_results: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Extracted content results from Step 1")
    
    # Enriched chunks
    enriched_chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Enriched chunks with metadata")
    
    # Enrichment statistics
    total_enriched_chunks: int = Field(default=0, description="Total number of chunks enriched")
    
    # Enrichment results
    enrichment_results: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Results of enrichment process")
    
    # Processing metadata
    enrichment_summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Summary of enrichment process")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True 