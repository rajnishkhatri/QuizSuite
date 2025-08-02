"""
Base Nodes Module

Provides base classes and utilities for all LangGraph nodes.
"""

import logging
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from langsmith import traceable

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """
    Base class for all LangGraph nodes with tracing support.
    
    Provides common functionality:
    - Logging and error handling
    - State validation
    - Performance monitoring
    - LangSmith tracing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base node.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.node_name = self.get_node_name()
        logger.info(f"Initialized {self.node_name}")
    
    @abstractmethod
    def get_node_name(self) -> str:
        """Get the name of this node."""
        pass
    
    @abstractmethod
    def process(self, state):
        """Process the state and return updated state."""
        pass
    
    def validate_state(self, state) -> bool:
        """
        Validate the input state.
        
        Args:
            state: State to validate
            
        Returns:
            True if valid, False otherwise
        """
        if state is None:
            logger.error("State is None")
            return False
        
        # Add more validation as needed
        return True
    
    def log_performance(self, start_time: float, end_time: float, node_name: str):
        """
        Log performance metrics.
        
        Args:
            start_time: Start time
            end_time: End time
            node_name: Name of the node
        """
        duration = end_time - start_time
        logger.info(f"{node_name} completed in {duration:.2f} seconds")


class TraceableNode(BaseNode):
    """
    Base class for nodes with LangSmith tracing support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the traceable node."""
        super().__init__(config)
    
    @traceable(run_type="chain", name="Document Processing Node")
    def process_with_tracing(self, state):
        """
        Process state with LangSmith tracing.
        
        Args:
            state: Input state
            
        Returns:
            Updated state
        """
        start_time = time.time()
        
        try:
            # Validate state
            if not self.validate_state(state):
                logger.error(f"State validation failed for {self.node_name}")
                return state
            
            # Process the state
            result = self.process(state)
            
            # Log performance
            end_time = time.time()
            self.log_performance(start_time, end_time, self.node_name)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {self.node_name}: {e}")
            # Return state with error information
            if hasattr(state, 'model_copy'):
                return state.model_copy(update={
                    'error': str(e),
                    'success': False,
                    'node_name': self.node_name
                })
            else:
                # Fallback for non-Pydantic states
                state['error'] = str(e)
                state['success'] = False
                state['node_name'] = self.node_name
                return state 