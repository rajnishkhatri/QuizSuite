"""
Chunking Strategy Module

This module provides different strategies for chunking documents, including
modality-aware chunking that extracts and processes images, figures, tables, and code.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .content_extractor import ContentExtractor

logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """
    Base class for document chunking strategies.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200):
        """
        Initialize the chunking strategy.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap_size: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        raise NotImplementedError("Subclasses must implement chunk_document")


class ModalityAwareChunkingStrategy(ChunkingStrategy):
    """
    Modality-aware chunking strategy that extracts and processes different content types.
    
    This strategy:
    - Splits documents into chunks based on content type
    - Extracts images, figures, tables, and code
    - Adds metadata for each content type
    - Handles multimodal content (text, images, tables, code)
    """
    
    def __init__(self, chunk_size: int = 1000, overlap_size: int = 200, 
                 extract_content: bool = True, output_dir: Optional[Path] = None):
        """
        Initialize the modality-aware chunking strategy.
        
        Args:
            chunk_size: Size of each text chunk in characters
            overlap_size: Overlap between text chunks in characters
            extract_content: Whether to extract images, figures, tables, and code
            output_dir: Directory to save extracted content
        """
        super().__init__(chunk_size, overlap_size)
        self.extract_content = extract_content
        self.output_dir = output_dir
        
        # Initialize content extractor if extraction is enabled
        self.content_extractor = None
        if self.extract_content:
            self.content_extractor = ContentExtractor(output_dir)
    
    def chunk_document(self, document: Dict[str, Any], extracted_content: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk a document using modality-aware strategy.
        
        Args:
            document: Document to chunk
            extracted_content: Pre-extracted content (optional, will extract if not provided)
            
        Returns:
            List of document chunks with extracted content
        """
        logger.info(f"Starting modality-aware chunking for document: {document.get('path', 'unknown')}")
        
        chunks = []
        
        # Use provided extracted content or extract if not provided
        if extracted_content is None and self.extract_content and self.content_extractor:
            try:
                pdf_path = document.get('path', '')
                if pdf_path and Path(pdf_path).exists():
                    logger.info(f"Extracting content from: {pdf_path}")
                    extracted_content = self.content_extractor.extract_all_content(pdf_path)
                    
                    # Save extracted content if output directory is specified
                    if self.output_dir:
                        extracted_content = self.content_extractor.save_extracted_content(
                            extracted_content, self.output_dir
                        )
                    
                    # Generate extraction summary
                    extraction_summary = self.content_extractor.get_extraction_summary(extracted_content)
                    logger.info(f"Content extraction summary: {extraction_summary}")
                else:
                    logger.warning(f"PDF path not found or invalid: {pdf_path}")
            except Exception as e:
                logger.error(f"Error during content extraction: {e}")
                extracted_content = None
        
        # Create text chunks
        text_chunks = self._create_text_chunks(document)
        
        # Create content-specific chunks
        content_chunks = self._create_content_chunks(extracted_content) if extracted_content else []
        
        # Combine all chunks
        chunks.extend(text_chunks)
        chunks.extend(content_chunks)
        
        logger.info(f"Created {len(chunks)} total chunks ({len(text_chunks)} text, {len(content_chunks)} content)")
        
        return chunks
    
    def _create_text_chunks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create text-based chunks from the document.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of text chunks
        """
        text_chunks = []
        content = document.get('content', '')
        
        if not content:
            logger.warning("No content found in document for text chunking")
            return text_chunks
        
        # Split content into chunks
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(content):
                # Look for sentence endings
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                for ending in sentence_endings:
                    pos = content.rfind(ending, start, end)
                    if pos > start:
                        end = pos + len(ending)
                        break
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'type': 'text',
                    'content': chunk_text,
                    'start_index': start,
                    'end_index': end,
                    'chunk_index': chunk_index,
                    'length': len(chunk_text),
                    'metadata': {
                        'chunk_size': self.chunk_size,
                        'overlap_size': self.overlap_size,
                        'modality': 'text'
                    }
                }
                text_chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = end - self.overlap_size
            if start >= len(content):
                break
        
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks
    
    def _create_content_chunks(self, extracted_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunks for extracted content (images, figures, tables, code).
        
        Args:
            extracted_content: Extracted content dictionary
            
        Returns:
            List of content chunks
        """
        content_chunks = []
        chunk_index = 0
        
        # Create image chunks
        for i, image in enumerate(extracted_content.get('images', [])):
            chunk = {
                'type': 'image',
                'content': image.get('extracted_text', ''),
                'chunk_index': chunk_index,
                'metadata': {
                    'page_number': image.get('page_number', 0),
                    'image_index': image.get('image_index', 0),
                    'width': image.get('width', 0),
                    'height': image.get('height', 0),
                    'ocr_success': image.get('ocr_success', False),
                    'bbox': image.get('bbox', None),
                    'modality': 'image',
                    'size_bytes': image.get('size_bytes', 0)
                }
            }
            content_chunks.append(chunk)
            chunk_index += 1
        
        # Create table chunks
        for i, table in enumerate(extracted_content.get('tables', [])):
            # Convert table data to text representation
            table_text = self._table_to_text(table.get('data', []))
            
            chunk = {
                'type': 'table',
                'content': table_text,
                'chunk_index': chunk_index,
                'metadata': {
                    'page_number': table.get('page_number', 0),
                    'table_index': table.get('table_index', 0),
                    'rows': table.get('rows', 0),
                    'columns': table.get('columns', 0),
                    'bbox': table.get('bbox', None),
                    'modality': 'table',
                    'total_cells': table.get('total_cells', 0),
                    'non_empty_cells': table.get('non_empty_cells', 0)
                }
            }
            content_chunks.append(chunk)
            chunk_index += 1
        
        # Create code chunks
        for i, code in enumerate(extracted_content.get('code_chunks', [])):
            chunk = {
                'type': 'code',
                'content': code.get('content', ''),
                'chunk_index': chunk_index,
                'metadata': {
                    'page_number': code.get('page_number', 0),
                    'chunk_index': code.get('chunk_index', 0),
                    'code_type': code.get('type', 'unknown'),
                    'lines': code.get('lines', 0),
                    'characters': code.get('characters', 0),
                    'modality': 'code'
                }
            }
            content_chunks.append(chunk)
            chunk_index += 1
        
        # Create figure chunks
        for i, figure in enumerate(extracted_content.get('figures', [])):
            chunk = {
                'type': 'figure',
                'content': figure.get('extracted_text', ''),
                'chunk_index': chunk_index,
                'metadata': {
                    'page_number': figure.get('page_number', 0),
                    'figure_type': figure.get('figure_type', 'unknown'),
                    'width': figure.get('width', 0),
                    'height': figure.get('height', 0),
                    'ocr_success': figure.get('ocr_success', False),
                    'bbox': figure.get('bbox', None),
                    'modality': 'figure',
                    'size_bytes': figure.get('size_bytes', 0)
                }
            }
            content_chunks.append(chunk)
            chunk_index += 1
        
        logger.info(f"Created {len(content_chunks)} content chunks")
        return content_chunks
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """
        Convert table data to text representation.
        
        Args:
            table_data: Table data as list of lists
            
        Returns:
            Text representation of the table
        """
        if not table_data:
            return ""
        
        text_lines = []
        text_lines.append("Table Content:")
        text_lines.append("=" * 50)
        
        for row in table_data:
            row_text = " | ".join([str(cell) for cell in row if cell])
            if row_text.strip():
                text_lines.append(row_text)
        
        return "\n".join(text_lines)
    
    def get_chunking_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the chunking results.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_chunks': len(chunks),
            'chunks_by_type': {},
            'chunks_by_modality': {},
            'content_stats': {
                'total_text_length': 0,
                'total_images': 0,
                'total_tables': 0,
                'total_code_chunks': 0,
                'total_figures': 0
            }
        }
        
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            modality = chunk.get('metadata', {}).get('modality', 'unknown')
            
            # Count by type
            summary['chunks_by_type'][chunk_type] = summary['chunks_by_type'].get(chunk_type, 0) + 1
            
            # Count by modality
            summary['chunks_by_modality'][modality] = summary['chunks_by_modality'].get(modality, 0) + 1
            
            # Update content stats
            if chunk_type == 'text':
                summary['content_stats']['total_text_length'] += len(chunk.get('content', ''))
            elif chunk_type == 'image':
                summary['content_stats']['total_images'] += 1
            elif chunk_type == 'table':
                summary['content_stats']['total_tables'] += 1
            elif chunk_type == 'code':
                summary['content_stats']['total_code_chunks'] += 1
            elif chunk_type == 'figure':
                summary['content_stats']['total_figures'] += 1
        
        return summary


class SimpleChunkingStrategy(ChunkingStrategy):
    """
    Simple chunking strategy that only splits text content.
    """
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using simple text-based strategy.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        logger.info(f"Starting simple chunking for document: {document.get('path', 'unknown')}")
        
        return self._create_text_chunks(document)
    
    def _create_text_chunks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create text-based chunks from the document.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of text chunks
        """
        text_chunks = []
        content = document.get('content', '')
        
        if not content:
            logger.warning("No content found in document for text chunking")
            return text_chunks
        
        # Split content into chunks
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(content):
                # Look for sentence endings
                sentence_endings = ['. ', '! ', '? ', '\n\n']
                for ending in sentence_endings:
                    pos = content.rfind(ending, start, end)
                    if pos > start:
                        end = pos + len(ending)
                        break
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'type': 'text',
                    'content': chunk_text,
                    'start_index': start,
                    'end_index': end,
                    'chunk_index': chunk_index,
                    'length': len(chunk_text),
                    'metadata': {
                        'chunk_size': self.chunk_size,
                        'overlap_size': self.overlap_size,
                        'modality': 'text'
                    }
                }
                text_chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = end - self.overlap_size
            if start >= len(content):
                break
        
        logger.info(f"Created {len(text_chunks)} text chunks")
        return text_chunks 