"""
Base State Module

This module defines the base state classes for LangGraph states.
"""

from typing import Optional, Dict, Any, Annotated
from pydantic import BaseModel, Field
from ..utils.reducers import dict_merge_reducer


class BaseState(BaseModel):
    """
    Base state class for LangGraph states.
    
    This class provides common functionality for all states in the pipeline.
    """
    
    # Common state attributes
    error: Optional[str] = Field(default=None, description="Error message if any")
    success: bool = Field(default=True, description="Whether the operation was successful")
    
    # Processing metadata
    node_name: Optional[str] = Field(default=None, description="Name of the current node")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    
    # Configuration
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration used")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class BaseNodeState(BaseState):
    """
    Base state with only common fields needed across all nodes.
    
    This state contains only the essential fields that are shared between nodes:
    - Configuration
    - Error handling
    - Success status
    - Node identification
    - Processing metadata
    """
    
    # Processing metadata
    total_processing_time: Optional[float] = Field(None, description="Total processing time")
    pipeline_success: bool = Field(default=False, description="Overall pipeline success") 