"""
Custom Reducers for LangGraph State Management

This module provides custom reducers for managing state in LangGraph applications.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def first_value_reducer(a: Any, b: Any) -> Any:
    """
    Custom reducer that always returns the first value.
    This ensures only one value is used in the pipeline for any field.
    
    Args:
        a: First value (any type)
        b: Second value (ignored)
        
    Returns:
        The first value (a)
    """
    return a  # Always return the first value


def dict_merge_reducer(a: Dict, b: Dict) -> Dict:
    """
    Custom reducer that merges two dictionaries.
    This allows multiple nodes to update the same dictionary field.
    
    Args:
        a: First dictionary
        b: Second dictionary
        
    Returns:
        Merged dictionary
    """
    if not isinstance(a, dict):
        a = {}
    if not isinstance(b, dict):
        b = {}
    return {**a, **b}


def list_merge_reducer(a: List, b: List) -> List:
    """
    Custom reducer that merges two lists.
    This allows multiple nodes to update the same list field.
    
    Args:
        a: First list
        b: Second list
        
    Returns:
        Merged list
    """
    if not isinstance(a, list):
        a = []
    if not isinstance(b, list):
        b = []
    return a + b 