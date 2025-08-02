"""
End Node State

State for the end node of the pipeline.
"""

from typing import Dict, Any, Optional, Annotated
from pydantic import Field

from .storage_state import StorageNodeState
from ..utils.reducers import dict_merge_reducer


class EndNodeState(StorageNodeState):
    """State for the end node of the pipeline."""
    final_summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(default_factory=dict, description="Final summary of the pipeline")
    pipeline_success: bool = Field(default=False, description="Whether the pipeline was successful")
    total_processing_time: Optional[float] = Field(default=None, description="Total processing time") 