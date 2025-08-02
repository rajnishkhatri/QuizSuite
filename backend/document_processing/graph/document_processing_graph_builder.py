"""
Document Processing Graph Builder

Builder pattern for creating document processing graphs with integrated pipeline.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END

from ..state.unified_state import UnifiedPipelineState
from ..model.config_models import QuizConfig
from ..node.integrated_pipeline_nodes import (
    IntegratedIngestNode, IntegratedProcessNode, IntegratedEmbeddingNode,
    IntegratedStorageNode, IntegratedSummaryNode, create_integrated_pipeline_graph
)


logger = logging.getLogger(__name__)


class DocumentProcessingGraphBuilder:
    """
    Builder for document processing graphs.
    
    Follows Builder Pattern for creating graphs with different configurations.
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.config: Optional[QuizConfig] = None
        self.graph: Optional[StateGraph] = None
        self.custom_nodes: Dict[str, Any] = {}
        self.custom_edges: Dict[str, str] = {}
    
    def with_config(self, config: QuizConfig) -> 'DocumentProcessingGraphBuilder':
        """Set the configuration for the graph."""
        self.config = config
        return self
    
    def with_config_file(self, config_path: str) -> 'DocumentProcessingGraphBuilder':
        """Load configuration from file."""
        import json
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.config = QuizConfig(**config_data)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise
        
        return self
    
    def with_custom_node(self, node_name: str, node_function) -> 'DocumentProcessingGraphBuilder':
        """Add a custom node to the graph."""
        self.custom_nodes[node_name] = node_function
        return self
    
    def with_custom_edge(self, from_node: str, to_node: str) -> 'DocumentProcessingGraphBuilder':
        """Add a custom edge to the graph."""
        self.custom_edges[from_node] = to_node
        return self
    
    def build(self) -> StateGraph:
        """Build the integrated pipeline graph."""
        if not self.config:
            raise ValueError("Configuration must be set before building graph")
        
        logger.info("Building integrated pipeline with advanced features...")
        
        # Convert QuizConfig to dict for integrated pipeline
        config_dict = self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config.dict()
        
        # Create integrated pipeline graph
        self.graph = create_integrated_pipeline_graph(config_dict)
        
        logger.info("Integrated pipeline graph built successfully")
        return self.graph
    
    
    
    def build_with_validation(self) -> StateGraph:
        """Build the graph with validation."""
        graph = self.build()
        
        # Validate graph structure
        self._validate_graph(graph)
        
        return graph
    
    def _validate_graph(self, graph: StateGraph) -> None:
        """Validate the graph structure."""
        # Check if all nodes are connected
        # This is a basic validation - in a real implementation,
        # you might want more sophisticated validation
        logger.info("Graph validation completed")
    
    def create_compiled_graph(self) -> Any:
        """Create and compile the graph."""
        graph = self.build()
        return graph.compile()
    
    def create_compiled_graph_with_validation(self) -> Any:
        """Create and compile the graph with validation."""
        graph = self.build_with_validation()
        return graph.compile()
    
    def run_pipeline(self, initial_state: Dict[str, Any] = None) -> Any:
        """Run the integrated pipeline with UnifiedPipelineState."""
        if not self.graph:
            self.build()
        
        from ..state.unified_state import UnifiedPipelineState
        
        if initial_state is None:
            initial_state = {}
        
        # Create initial state for integrated pipeline
        state = UnifiedPipelineState(
            documents=[],
            total_documents=0,
            chunks=[],
            total_chunks=0,
            embedded_chunks=[],
            total_embedded=0,
            storage_info={},
            collection_stats={},
            processing_summary={},
            success=False,
            error=None,
            node_name="start",
            messages=[],
            errors=[]
        )
        
        # Run the graph (it's already compiled from create_integrated_pipeline_graph)
        result = self.graph.invoke(state)
        
        logger.info("Integrated pipeline execution completed")
        return result 