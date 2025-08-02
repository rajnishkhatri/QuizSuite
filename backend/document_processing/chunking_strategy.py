"""
Modality-Aware Chunking Strategy Module

Implements intelligent chunking that preserves different content modalities (text, images, tables).
Follows Strategy Pattern for different chunking approaches.
"""

import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import re

from .models import DocumentChunk, ModalityType, ProcessedDocument


logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """
    Abstract base class for chunking strategies.
    
    Follows Strategy Pattern - allows different chunking approaches.
    """
    
    @abstractmethod
    def chunk_document(self, document: ProcessedDocument, chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Chunk a document according to this strategy."""
        pass


class ModalityAwareChunkingStrategy(ChunkingStrategy):
    """
    Implements modality-aware chunking that preserves different content types.
    
    This strategy recognizes and preserves different modalities (text, images, tables)
    while maintaining semantic coherence.
    """
    
    def __init__(self) -> None:
        """Initialize the modality-aware chunking strategy."""
        self.modality_detectors = self._initialize_modality_detectors()
    
    def _initialize_modality_detectors(self) -> Dict[str, callable]:
        """Initialize detectors for different content modalities."""
        return {
            "table": self._detect_table,
            "image": self._detect_image,
            "code": self._detect_code,
            "figure": self._detect_figure,
        }
    
    def chunk_document(self, document: ProcessedDocument, chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Chunk document while preserving modality boundaries."""
        chunks = []
        
        try:
            raw_content = self._extract_raw_content(document)
            modality_segments = self._segment_by_modality(raw_content)
            
            for segment in modality_segments:
                segment_chunks = self._chunk_segment(segment, chunk_size, overlap_size)
                chunks.extend(segment_chunks)
            
            self._assign_chunk_metadata(chunks, document)
            logger.info(f"Created {len(chunks)} chunks for document {document.document_id}")
            
        except Exception as e:
            logger.error(f"Error chunking document {document.document_id}: {e}")
            raise
        
        return chunks
    
    def _extract_raw_content(self, document: ProcessedDocument) -> str:
        """Extract raw content from document based on type."""
        if document.document_type.value == "pdf":
            return self._extract_pdf_content(document.file_path)
        else:
            raise ValueError(f"Unsupported document type: {document.document_type}")
    
    def _extract_pdf_content(self, file_path) -> str:
        """Extract text content from PDF file."""
        try:
            import fitz
            
            doc = fitz.open(file_path)
            content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                content += page.get_text()
                content += "\n"  # Add page separator
            
            doc.close()
            return content
            
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise
    
    def _segment_by_modality(self, content: str) -> List[Dict[str, Any]]:
        """Segment content by modality type."""
        segments = []
        lines = content.split('\n')
        current_segment = {"type": ModalityType.TEXT, "content": "", "start_line": 0}
        
        for line_num, line in enumerate(lines):
            detected_modality = self._detect_line_modality(line)
            
            if detected_modality != current_segment["type"]:
                # Save current segment if it has content
                if current_segment["content"].strip():
                    current_segment["end_line"] = line_num - 1
                    segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    "type": detected_modality,
                    "content": line,
                    "start_line": line_num
                }
            else:
                current_segment["content"] += "\n" + line
        
        # Add final segment
        if current_segment["content"].strip():
            current_segment["end_line"] = len(lines) - 1
            segments.append(current_segment)
        
        return segments
    
    def _detect_line_modality(self, line: str) -> ModalityType:
        """Detect the modality type of a single line."""
        line = line.strip()
        
        if not line:
            return ModalityType.TEXT
        
        # Check for table patterns
        if self._detect_table(line):
            return ModalityType.TABLE
        
        # Check for code patterns
        if self._detect_code(line):
            return ModalityType.CODE
        
        # Check for image/figure references
        if self._detect_image(line):
            return ModalityType.IMAGE
        
        if self._detect_figure(line):
            return ModalityType.FIGURE
        
        return ModalityType.TEXT
    
    def _detect_table(self, line: str) -> bool:
        """Detect if line contains table content."""
        # Look for table indicators like multiple tabs, pipes, or consistent spacing
        if '|' in line and line.count('|') >= 2:
            return True
        
        # Check for tab-separated values
        if '\t' in line and line.count('\t') >= 2:
            return True
        
        # Check for consistent spacing patterns (table-like)
        if re.match(r'^\s*\S+\s{2,}\S+', line):
            return True
        
        return False
    
    def _detect_image(self, line: str) -> bool:
        """Detect if line contains image references."""
        image_patterns = [
            r'\.(jpg|jpeg|png|gif|bmp|tiff|svg)\b',
            r'\[.*?\]\(.*?\.(jpg|jpeg|png|gif|bmp|tiff|svg)\)',
            r'image:',
            r'img:',
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in image_patterns)
    
    def _detect_figure(self, line: str) -> bool:
        """Detect if line contains figure references."""
        figure_patterns = [
            r'figure\s+\d+',
            r'fig\.\s*\d+',
            r'fig\s+\d+',
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in figure_patterns)
    
    def _detect_code(self, line: str) -> bool:
        """Detect if line contains code content."""
        code_patterns = [
            r'^\s*(def|class|import|from|if|for|while|try|except|with)\s',
            r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]\s*',
            r'^\s*[{}()\[\]]\s*$',
            r'^\s*#.*$',  # Comments
        ]
        
        return any(re.search(pattern, line) for pattern in code_patterns)
    
    def _chunk_segment(self, segment: Dict[str, Any], chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Chunk a single modality segment."""
        chunks = []
        content = segment["content"]
        modality = segment["type"]
        
        if modality == ModalityType.TABLE:
            # Keep tables as single chunks to preserve structure
            chunks.append(self._create_chunk(content, modality, 0))
        elif modality == ModalityType.IMAGE:
            # Keep image references as single chunks
            chunks.append(self._create_chunk(content, modality, 0))
        elif modality == ModalityType.FIGURE:
            # Keep figure references as single chunks
            chunks.append(self._create_chunk(content, modality, 0))
        else:
            # For text and code, apply sliding window chunking
            chunks = self._sliding_window_chunk(content, chunk_size, overlap_size, modality)
        
        return chunks
    
    def _sliding_window_chunk(self, content: str, chunk_size: int, overlap_size: int, modality: ModalityType) -> List[DocumentChunk]:
        """Apply sliding window chunking to text content."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at word boundaries
            if end < len(content):
                # Look for the last space before the end
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append(self._create_chunk(chunk_content, modality, chunk_index))
                chunk_index += 1
            
            start = end - overlap_size
            if start >= len(content):
                break
        
        return chunks
    
    def _create_chunk(self, content: str, modality: ModalityType, chunk_index: int) -> DocumentChunk:
        """Create a document chunk with proper metadata."""
        return DocumentChunk(
            content=content,
            chunk_id=f"chunk_{chunk_index}_{modality.value}",
            document_id="",  # Will be set later
            modality=modality,
            chunk_index=chunk_index,
            metadata={"modality": modality.value}
        )
    
    def _assign_chunk_metadata(self, chunks: List[DocumentChunk], document: ProcessedDocument) -> None:
        """Assign document metadata to chunks."""
        for chunk in chunks:
            chunk.document_id = document.document_id
            chunk.metadata.update({
                "source_file": str(document.file_path),
                "document_type": document.document_type.value,
                "processing_status": document.processing_status
            }) 