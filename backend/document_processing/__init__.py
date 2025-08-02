"""
Document Processing Package

Handles document ingestion, processing, and chunking for the quiz generation system.
"""

from .model import (
    DocumentType,
    ModalityType,
    DocumentChunk,
    ProcessedDocument,
    ChromaDBSettings,
    CategoryConfig,
    AutoTopicDistributionSettings,
    CacheSettings,
    OutputSettings,
    PromptSettings,
    QuizConfig,
    DocumentProcessingState,
    IngestRequest
)

from .state import (
    BaseState,
    IngestNodeState,
    ProcessNodeState,
    EndNodeState
)

from .node import (
    DocumentProcessingNodes,
    DocumentProcessingGraph,
    create_document_processing_graph
)

from .graph import (
    DocumentProcessingGraphBuilder
)

__all__ = [
    # Models
    "DocumentType",
    "ModalityType", 
    "DocumentChunk",
    "ProcessedDocument",
    "ChromaDBSettings",
    "CategoryConfig",
    "AutoTopicDistributionSettings",
    "CacheSettings",
    "OutputSettings",
    "PromptSettings",
    "QuizConfig",
    "DocumentProcessingState",
    "IngestRequest",
    
    # States
    "BaseState",
    "IngestNodeState",
    "ProcessNodeState", 
    "EndNodeState",
    
    # Nodes
    "DocumentProcessingNodes",
    "DocumentProcessingGraph",
    "create_document_processing_graph",
    
    # Graphs
    "DocumentProcessingGraphBuilder"
] 