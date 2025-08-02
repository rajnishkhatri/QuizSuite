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
    UnifiedPipelineState
)

from .node import (
    IntegratedIngestNode,
    IntegratedProcessNode,
    IntegratedEmbeddingNode,
    IntegratedStorageNode,
    IntegratedSummaryNode,
    create_integrated_pipeline_graph
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
    "UnifiedPipelineState",
    
    # Nodes
    "IntegratedIngestNode",
    "IntegratedProcessNode",
    "IntegratedEmbeddingNode",
    "IntegratedStorageNode",
    "IntegratedSummaryNode",
    "create_integrated_pipeline_graph",
    
    # Graphs
    "DocumentProcessingGraphBuilder"
] 