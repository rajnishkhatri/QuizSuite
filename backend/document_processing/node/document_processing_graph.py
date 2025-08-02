"""
Document Processing Graph

LangGraph graph for document processing pipeline.
"""

import logging
import json
from typing import Dict, Any

from langgraph.graph import StateGraph, END

from ..state.ingest_state import IngestNodeState
from ..state.embed_state import EmbedNodeState
from ..model.config_models import QuizConfig
from .document_processing_nodes import DocumentProcessingNodes


logger = logging.getLogger(__name__)


class DocumentProcessingGraph:
    """
    LangGraph graph for document processing pipeline.
    
    Implements the graph: START → Ingest → Process → Embed & Store → END
    """
    
    def __init__(self, config: QuizConfig):
        """Initialize the document processing graph."""
        self.config = config
        self.nodes = DocumentProcessingNodes(config)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create the graph
        workflow = StateGraph(IngestNodeState)
        
        # Add nodes
        workflow.add_node("ingest", self.nodes.ingest_documents_node)
        workflow.add_node("process", self.nodes.process_documents_node)
        workflow.add_node("embed_and_store", self.nodes.embed_and_store_node)
        workflow.add_node("end", self.nodes.end_node)
        
        # Define the flow
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "process")
        workflow.add_edge("process", "embed_and_store")
        workflow.add_edge("embed_and_store", "end")
        workflow.add_edge("end", END)
        
        return workflow
    
    def compile(self):
        """Compile the graph for execution."""
        return self.graph.compile()
    
    def run(self, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the document processing pipeline."""
        if initial_state is None:
            initial_state = {}
        
        # Create initial state
        state = IngestNodeState(**initial_state)
        
        # Compile and run the graph
        compiled_graph = self.compile()
        result = compiled_graph.invoke(state)
        
        return result


def create_document_processing_graph(config_path: str = "config/quiz_config.json") -> DocumentProcessingGraph:
    """
    Factory function to create document processing graph from config file.
    
    Args:
        config_path: Path to the quiz configuration file
        
    Returns:
        DocumentProcessingGraph instance
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = QuizConfig(**config_data)
        
        # Create and return graph
        return DocumentProcessingGraph(config)
        
    except Exception as e:
        logger.error(f"Error creating document processing graph: {e}")
        raise 