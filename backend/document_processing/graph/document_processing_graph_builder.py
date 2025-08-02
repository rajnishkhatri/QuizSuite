"""
Document Processing Graph Builder

Builder pattern for creating document processing graphs with different configurations.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END

from ..state.ingest_state import IngestNodeState
from ..model.config_models import QuizConfig
from ..node.document_processing_nodes import DocumentProcessingNodes


logger = logging.getLogger(__name__)


class DocumentProcessingGraphBuilder:
    """
    Builder for document processing graphs.
    
    Follows Builder Pattern for creating graphs with different configurations.
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.config: Optional[QuizConfig] = None
        self.nodes: Optional[DocumentProcessingNodes] = None
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
        """Build the document processing graph."""
        if not self.config:
            raise ValueError("Configuration must be set before building graph")
        
        # Create nodes
        self.nodes = DocumentProcessingNodes(self.config)
        
        # Create the graph
        self.graph = StateGraph(IngestNodeState)
        
        # Add standard nodes
        self.graph.add_node("ingest", self.nodes.ingest_documents_node)
        self.graph.add_node("process", self.nodes.process_documents_node)
        self.graph.add_node("end", self.nodes.end_node)
        
        # Add custom nodes
        for node_name, node_function in self.custom_nodes.items():
            self.graph.add_node(node_name, node_function)
        
        # Set entry point
        self.graph.set_entry_point("ingest")
        
        # Add standard edges
        self.graph.add_edge("ingest", "process")
        self.graph.add_edge("process", "end")
        self.graph.add_edge("end", END)
        
        # Add custom edges
        for from_node, to_node in self.custom_edges.items():
            self.graph.add_edge(from_node, to_node)
        
        logger.info("Document processing graph built successfully")
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