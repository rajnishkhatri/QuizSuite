"""
Clean Node State

State for text cleaning node (Step 2).
"""

from typing import List, Dict, Any, Annotated, Optional
from pydantic import Field
from .base_state import BaseState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer


class CleanNodeState(BaseState):
    """
    State model for text cleaning node (Step 2).
    
    This state contains:
    - Documents to clean
    - Cleaned documents
    - Cleaning statistics
    """
    
    # Documents to clean
    documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Documents to clean")
    
    # Cleaned documents
    cleaned_documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(default_factory=list, description="Cleaned documents")
    
    # Cleaning statistics
    total_documents_cleaned: int = Field(default=0, description="Total number of documents cleaned")
    
    # Processing metadata
    cleaning_summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Summary of cleaning process")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True 