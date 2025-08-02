"""
Step-by-Step Pipeline Nodes

This module implements individual nodes for each step of the document processing pipeline:
1. Extract content (images, figures, tables, code)
2. Clean and preprocess text
3. Create chunks using modality-aware strategy
4. Enrich chunks with metadata

Each step is implemented as a separate LangGraph node for explicit control and monitoring.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from langgraph.graph import StateGraph, END
from langsmith import traceable
from .base_nodes import TraceableNode
from ..processor.content_extractor import ContentExtractor
from ..processor.chunking_strategy import ModalityAwareChunkingStrategy
from ..processor.metadata_enricher import MetadataEnricher
from ..processor.text_cleaner import ModalityAwareTextCleaner
from ..state.ingest_state import IngestNodeState
from ..state.process_state import ProcessNodeState
from ..state.extract_state import ExtractNodeState
from ..state.clean_state import CleanNodeState
from ..state.chunk_state import ChunkNodeState
from ..state.enrich_state import EnrichNodeState

logger = logging.getLogger(__name__)


class ExtractContentNode(TraceableNode):
    """
    Extract content (images, figures, tables, code) from documents with tracing.
    
    This node handles:
    - Image extraction and OCR
    - Table extraction and conversion
    - Figure identification and extraction
    - Code block detection and extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the content extraction node."""
        super().__init__(config)
        
        # Initialize content extractor
        output_dir = Path("output/extracted_content")
        self.content_extractor = ContentExtractor(output_dir)
    
    @traceable(run_type="chain", name="Content Extraction Node")
    def process(self, state: ExtractNodeState) -> ExtractNodeState:
        """
        Extract content from documents.
        
        Args:
            state: Current state with documents
            
        Returns:
            Updated state with extracted content
        """
        logger.info("Starting content extraction")
        
        try:
            documents = state.documents
            if not documents:
                logger.warning("No documents to extract content from")
                return state.model_copy(update={
                    'error': "No documents to extract content from",
                    'success': False,
                    'node_name': 'extract_content_node'
                })
            
            # Extract content from each document
            extracted_content_results = []
            
            for doc in documents:
                try:
                    pdf_path = doc.get('path', '')
                    if pdf_path and Path(pdf_path).exists():
                        logger.info(f"Extracting content from: {pdf_path}")
                        
                        # Extract all content types
                        extracted_content = self.content_extractor.extract_all_content(pdf_path)
                        
                        # Save extracted content
                        saved_content = self.content_extractor.save_extracted_content(
                            extracted_content, self.content_extractor.output_dir
                        )
                        
                        # Get extraction summary
                        extraction_summary = self.content_extractor.get_extraction_summary(extracted_content)
                        
                        extracted_content_results.append({
                            'document_path': pdf_path,
                            'extracted_content': extracted_content,
                            'saved_content': saved_content,
                            'extraction_summary': extraction_summary
                        })
                        
                        logger.info(f"Content extraction completed for {pdf_path}: {extraction_summary}")
                        
                    else:
                        logger.warning(f"PDF path not found or invalid: {pdf_path}")
                        
                except Exception as e:
                    logger.error(f"Error extracting content from {doc.get('path', 'unknown')}: {e}")
            
            # Update state
            updated_state = state.model_copy(update={
                'extracted_content_results': extracted_content_results,
                'total_documents_extracted': len(extracted_content_results),
                'success': True,
                'node_name': 'extract_content_node',
                'processing_time': time.time()
            })
            
            logger.info(f"Content extraction completed: Extracted content from {len(extracted_content_results)} documents")
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in content extraction: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'extract_content_node'
            })
    
    def get_node_name(self) -> str:
        return "extract_content_node"


class CleanTextNode(TraceableNode):
    """
    Clean and preprocess text from documents with tracing.
    
    This node handles:
    - Text normalization
    - Whitespace cleaning
    - Special character handling
    - Text preprocessing for better chunking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the text cleaning node."""
        super().__init__(config)
        
        # Initialize text cleaner
        self.text_cleaner = ModalityAwareTextCleaner()
    
    @traceable(run_type="chain", name="Text Cleaning Node")
    def process(self, state: CleanNodeState) -> CleanNodeState:
        """
        Clean and preprocess text from documents.
        
        Args:
            state: Current state with documents and extracted content
            
        Returns:
            Updated state with cleaned documents
        """
        logger.info("Starting text cleaning and preprocessing")
        
        try:
            documents = state.documents
            if not documents:
                logger.warning("No documents to clean")
                return state.model_copy(update={
                    'error': "No documents to clean",
                    'success': False,
                    'node_name': 'clean_text_node'
                })
            
            # Clean each document
            cleaned_documents = []
            
            for doc in documents:
                try:
                    # Clean the document
                    cleaned_doc = self._clean_document(doc)
                    cleaned_documents.append(cleaned_doc)
                    
                    logger.info(f"Cleaned document: {doc.get('path', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning document {doc.get('path', 'unknown')}: {e}")
            
            # Update state
            updated_state = state.model_copy(update={
                'cleaned_documents': cleaned_documents,
                'total_documents_cleaned': len(cleaned_documents),
                'success': True,
                'node_name': 'clean_text_node',
                'processing_time': time.time()
            })
            
            logger.info(f"Text cleaning completed: Cleaned {len(cleaned_documents)} documents")
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in text cleaning: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'clean_text_node'
            })
    
    def _clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and preprocess a single document."""
        cleaned_document = document.copy()
        content = document.get('content', '')
        
        if content:
            # Clean the text
            cleaned_content = self.text_cleaner.clean_text(content)
            cleaned_document['content'] = cleaned_content
            cleaned_document['original_length'] = len(content)
            cleaned_document['cleaned_length'] = len(cleaned_content)
        
        return cleaned_document
    
    def get_node_name(self) -> str:
        return "clean_text_node"


class CreateChunksNode(TraceableNode):
    """
    Create chunks using modality-aware strategy with tracing.
    
    This node handles:
    - Modality-aware chunking
    - Content type identification
    - Chunk size and overlap management
    - Chunk metadata assignment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the chunking node."""
        super().__init__(config)
        
        # Initialize chunking strategy
        chunk_size = config.get('chunk_size', 1000)
        overlap_size = config.get('overlap_size', 200)
        output_dir = Path("output/chunks")
        
        self.chunking_strategy = ModalityAwareChunkingStrategy(
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            extract_content=True,
            output_dir=output_dir
        )
    
    @traceable(run_type="chain", name="Chunking Node")
    def process(self, state: ChunkNodeState) -> ChunkNodeState:
        """
        Create chunks using modality-aware strategy.
        
        Args:
            state: Current state with cleaned documents and extracted content
            
        Returns:
            Updated state with chunks
        """
        logger.info("Starting modality-aware chunking")
        
        try:
            cleaned_documents = state.cleaned_documents
            extracted_content_results = state.extracted_content_results
            
            if not cleaned_documents:
                logger.warning("No cleaned documents to chunk")
                return state.model_copy(update={
                    'error': "No cleaned documents to chunk",
                    'success': False,
                    'node_name': 'create_chunks_node'
                })
            
            # Create chunks for each document
            all_chunks = []
            chunking_results = []
            
            for i, doc in enumerate(cleaned_documents):
                try:
                    # Get corresponding extracted content
                    extracted_content = None
                    if extracted_content_results and i < len(extracted_content_results):
                        extracted_content = extracted_content_results[i].get('extracted_content')
                    
                    # Create chunks using modality-aware strategy
                    chunks = self.chunking_strategy.chunk_document(doc, extracted_content)
                    
                    # Add document metadata to chunks
                    for chunk in chunks:
                        chunk['document_path'] = doc.get('path', 'unknown')
                        chunk['document_category'] = doc.get('category', 'unknown')
                    
                    all_chunks.extend(chunks)
                    chunking_results.append({
                        'document_path': doc.get('path', 'unknown'),
                        'chunks_count': len(chunks),
                        'chunks_by_type': self._count_chunks_by_type(chunks)
                    })
                    
                    logger.info(f"Created {len(chunks)} chunks for document: {doc.get('path', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error chunking document {doc.get('path', 'unknown')}: {e}")
            
            # Update state
            updated_state = state.model_copy(update={
                'chunks': all_chunks,
                'chunking_results': chunking_results,
                'total_chunks': len(all_chunks),
                'success': True,
                'node_name': 'create_chunks_node',
                'processing_time': time.time()
            })
            
            logger.info(f"Chunking completed: Created {len(all_chunks)} total chunks")
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in chunking: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'create_chunks_node'
            })
    
    def _count_chunks_by_type(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count chunks by type."""
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        return type_counts
    
    def get_node_name(self) -> str:
        return "create_chunks_node"


class EnrichChunksNode(TraceableNode):
    """
    Enrich chunks with metadata with tracing.
    
    This node handles:
    - Metadata enrichment
    - Content type metadata
    - Processing metadata
    - Quality metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the chunk enrichment node."""
        super().__init__(config)
        
        # Initialize metadata enricher
        self.metadata_enricher = MetadataEnricher()
    
    @traceable(run_type="chain", name="Chunk Enrichment Node")
    def process(self, state: EnrichNodeState) -> EnrichNodeState:
        """
        Enrich chunks with metadata.
        
        Args:
            state: Current state with chunks and extracted content
            
        Returns:
            Updated state with enriched chunks
        """
        logger.info("Starting chunk metadata enrichment")
        
        try:
            chunks = state.chunks
            extracted_content_results = state.extracted_content_results
            
            if not chunks:
                logger.warning("No chunks to enrich")
                return state.model_copy(update={
                    'error': "No chunks to enrich",
                    'success': False,
                    'node_name': 'enrich_chunks_node'
                })
            
            # Enrich each chunk with metadata
            enriched_chunks = []
            
            for chunk in chunks:
                try:
                    # Enrich chunk with metadata
                    enriched_chunk = self._enrich_chunk(chunk, extracted_content_results)
                    enriched_chunks.append(enriched_chunk)
                    
                except Exception as e:
                    logger.error(f"Error enriching chunk: {e}")
                    # Add basic metadata even if enrichment fails
                    chunk['metadata'] = chunk.get('metadata', {})
                    chunk['metadata']['enrichment_error'] = str(e)
                    enriched_chunks.append(chunk)
            
            # Update state
            updated_state = state.model_copy(update={
                'enriched_chunks': enriched_chunks,
                'total_enriched_chunks': len(enriched_chunks),
                'success': True,
                'node_name': 'enrich_chunks_node',
                'processing_time': time.time()
            })
            
            logger.info(f"Chunk enrichment completed: Enriched {len(enriched_chunks)} chunks")
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in chunk enrichment: {e}")
            return state.model_copy(update={
                'error': str(e),
                'success': False,
                'node_name': 'enrich_chunks_node'
            })
    
    def _enrich_chunk(self, chunk: Dict[str, Any], extracted_content_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enrich a single chunk with metadata."""
        enriched_chunk = chunk.copy()
        
        # Add processing metadata
        enriched_chunk['metadata'] = enriched_chunk.get('metadata', {})
        enriched_chunk['metadata'].update({
            'processed': True,
            'enrichment_step': 'completed',
            'chunk_size': len(chunk.get('content', '')),
            'chunk_type': chunk.get('type', 'unknown'),
            'modality': chunk.get('metadata', {}).get('modality', 'unknown')
        })
        
        # Add content-specific metadata
        chunk_type = chunk.get('type')
        if chunk_type == 'image':
            enriched_chunk['metadata'].update({
                'content_type': 'image',
                'has_ocr': chunk.get('metadata', {}).get('ocr_success', False),
                'image_dimensions': f"{chunk.get('metadata', {}).get('width', 0)}x{chunk.get('metadata', {}).get('height', 0)}"
            })
        elif chunk_type == 'table':
            enriched_chunk['metadata'].update({
                'content_type': 'table',
                'table_rows': chunk.get('metadata', {}).get('table_rows', 0),
                'table_columns': chunk.get('metadata', {}).get('table_columns', 0)
            })
        elif chunk_type == 'code':
            enriched_chunk['metadata'].update({
                'content_type': 'code',
                'code_type': chunk.get('metadata', {}).get('code_type', 'unknown'),
                'code_lines': chunk.get('metadata', {}).get('code_lines', 0)
            })
        else:
            enriched_chunk['metadata'].update({
                'content_type': 'text',
                'text_length': len(chunk.get('content', ''))
            })
        
        return enriched_chunk
    
    def get_node_name(self) -> str:
        return "enrich_chunks_node"


def create_step_by_step_pipeline_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create the step-by-step pipeline graph.
    
    Flow: START → Ingest → Extract → Clean → Chunk → Enrich → END
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled StateGraph
    """
    # Create the graph
    workflow = StateGraph(IngestNodeState)
    
    # Add nodes
    ingest_node = IntegratedIngestNode(config)
    extract_node = ExtractContentNode(config)
    clean_node = CleanTextNode(config)
    chunk_node = CreateChunksNode(config)
    enrich_node = EnrichChunksNode(config)
    
    workflow.add_node("ingest", ingest_node.process_with_tracing)
    workflow.add_node("extract", extract_node.process_with_tracing)
    workflow.add_node("clean", clean_node.process_with_tracing)
    workflow.add_node("chunk", chunk_node.process_with_tracing)
    workflow.add_node("enrich", enrich_node.process_with_tracing)
    
    # Define the flow
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "clean")
    workflow.add_edge("clean", "chunk")
    workflow.add_edge("chunk", "enrich")
    workflow.add_edge("enrich", END)
    
    # Compile the graph
    app = workflow.compile()
    
    logger.info("Step-by-step pipeline graph created successfully")
    return app 