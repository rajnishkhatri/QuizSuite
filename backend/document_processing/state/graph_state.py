"""
GraphDB State Module

This module defines state models for the GraphDB pipeline that integrates
with the existing unified pipeline state. GraphDB builds knowledge graphs
from extracted entities across different modalities (text, images, tables).
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import Field

from .base_state import BaseNodeState
from ..utils.reducers import list_merge_reducer, dict_merge_reducer, first_value_reducer


class GraphDBPipelineState(BaseNodeState):
    """
    Extended state for GraphDB pipeline.
    
    This state extends the base node state with GraphDB-specific fields
    for entity extraction, visual detection, and knowledge graph construction.
    GraphDB builds knowledge graphs from extracted entities across different modalities.
    """
    
    # Document processing fields (now part of GraphDBPipelineState)
    documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Raw documents from ingestion"
    )
    processed_documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Processed documents with chunks"
    )
    chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Document chunks for processing"
    )
    file_paths: Annotated[List[str], list_merge_reducer] = Field(
        default_factory=list, 
        description="File paths of processed documents"
    )
    
    # Processing statistics
    total_documents: Annotated[int, first_value_reducer] = Field(default=0, description="Total number of documents")
    total_chunks: Annotated[int, first_value_reducer] = Field(default=0, description="Total number of chunks")
    total_embedded: Annotated[int, first_value_reducer] = Field(default=0, description="Total number of embedded chunks")
    total_stored: Annotated[int, first_value_reducer] = Field(default=0, description="Total number of stored chunks")
    
    # Processing metadata
    processing_summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Processing summary and statistics"
    )
    modality_distribution: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Distribution of content modalities"
    )
    embedding_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Embedding generation statistics"
    )
    collection_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Collection storage statistics"
    )
    
    # Final results
    final_documents: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Final processed documents"
    )
    total_processed: Annotated[int, first_value_reducer] = Field(default=0, description="Total processed documents")
    summary: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Final pipeline summary"
    )
    
    # Messages and errors
    messages: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Processing messages and logs"
    )
    errors: Annotated[List[str], list_merge_reducer] = Field(
        default_factory=list, 
        description="Error messages from processing"
    )
    
    # Entity extraction results
    graphDB_text_entities: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Named entities, concepts, and keywords extracted from text"
    )
    llm_text_entities: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Entities extracted by LLM entity extraction"
    )
    llm_nodes: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Nodes created by LLM entity extraction"
    )
    llm_edges: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Edges created by LLM entity extraction"
    )
    graphDB_visual_entities: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Objects, visual concepts, and scene analysis from images"
    )
    graphDB_data_entities: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Column headers, key metrics, and data relationships from tables"
    )
    
    # Graph construction results
    graphDB_nodes: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Knowledge graph nodes"
    )
    graphDB_edges: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Knowledge graph edges/relationships"
    )
    graphDB_cross_modal_connections: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Connections between different modalities (text, image, table)"
    )
    
    # Graph enrichment results
    graphDB_enriched_nodes: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Nodes with enriched metadata and relationships"
    )
    graphDB_enriched_edges: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Enriched edges with relationship types (REFERENCES, DEPICTS, etc.)"
    )
    
    # Graph statistics
    graphDB_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Graph construction statistics and metrics"
    )
    
    # Graph database storage
    graphDB_storage_info: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Graph database storage information"
    )
    
    # Processing metadata
    graphDB_entity_extraction_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Entity extraction processing statistics"
    )
    graphDB_construction_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Graph construction processing statistics"
    )
    llm_extraction_metadata: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="LLM entity extraction metadata and statistics"
    )
    
    # Parallel processing fields for optimized LLM extraction
    filtered_chunks: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Chunks after quality filtering for LLM processing"
    )
    batches: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Batches of chunks for parallel LLM processing"
    )
    batch_metadata: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Batch processing metadata and statistics"
    )
    parallel_llm_results: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Results from parallel LLM entity extraction"
    )
    batch_results: Annotated[List[Dict[str, Any]], list_merge_reducer] = Field(
        default_factory=list, 
        description="Results from individual batch subgraph executions"
    )
    
    # Cache management fields
    cache_storage_result: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Cache storage results and metadata"
    )
    cache_info: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Cache information for test compatibility"
    )
    quality_filter_stats: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Quality filtering statistics and metadata"
    )
    incremental_graph_results: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Results from incremental graph building"
    )
    graph_metadata: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Metadata about the graph construction and updates"
    )
    
    # Structure-aware chunking results
    structure_aware_chunking_result: Annotated[Dict[str, Any], dict_merge_reducer] = Field(
        default_factory=dict, 
        description="Results from structure-aware chunking processing"
    ) 