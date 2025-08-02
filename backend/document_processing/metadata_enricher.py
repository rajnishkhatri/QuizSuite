"""
Metadata Enricher Module

Enriches document chunks with additional metadata for better retrieval and processing.
Follows Single Responsibility Principle - only responsible for metadata enrichment.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
import hashlib

from .models import DocumentChunk, ProcessedDocument


logger = logging.getLogger(__name__)


class MetadataEnricher:
    """
    Enriches document chunks with additional metadata.
    
    Follows Single Responsibility Principle - only handles metadata enrichment.
    """
    
    def __init__(self) -> None:
        """Initialize the metadata enricher."""
        self.enrichment_strategies = self._initialize_enrichment_strategies()
    
    def _initialize_enrichment_strategies(self) -> Dict[str, callable]:
        """Initialize different metadata enrichment strategies."""
        return {
            "content_hash": self._add_content_hash,
            "timestamp": self._add_timestamp,
            "length_metrics": self._add_length_metrics,
            "semantic_metadata": self._add_semantic_metadata,
            "modality_specific": self._add_modality_specific_metadata,
        }
    
    def enrich_chunks(self, chunks: List[DocumentChunk], document: ProcessedDocument) -> List[DocumentChunk]:
        """Enrich all chunks with additional metadata."""
        enriched_chunks = []
        
        for chunk in chunks:
            try:
                enriched_chunk = self._enrich_single_chunk(chunk, document)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.error(f"Error enriching chunk {chunk.chunk_id}: {e}")
                # Continue with other chunks even if one fails
                enriched_chunks.append(chunk)
        
        logger.info(f"Enriched {len(enriched_chunks)} chunks for document {document.document_id}")
        return enriched_chunks
    
    def _enrich_single_chunk(self, chunk: DocumentChunk, document: ProcessedDocument) -> DocumentChunk:
        """Enrich a single chunk with metadata."""
        enriched_metadata = chunk.metadata.copy()
        
        # Apply all enrichment strategies
        for strategy_name, strategy_func in self.enrichment_strategies.items():
            try:
                strategy_result = strategy_func(chunk, document)
                enriched_metadata.update(strategy_result)
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed for chunk {chunk.chunk_id}: {e}")
        
        # Create new chunk with enriched metadata
        return DocumentChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=enriched_metadata,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index
        )
    
    def _add_content_hash(self, chunk: DocumentChunk, document: ProcessedDocument) -> Dict[str, Any]:
        """Add content hash for deduplication and integrity checking."""
        content_hash = hashlib.sha256(chunk.content.encode('utf-8')).hexdigest()
        return {"content_hash": content_hash}
    
    def _add_timestamp(self, chunk: DocumentChunk, document: ProcessedDocument) -> Dict[str, Any]:
        """Add processing timestamp."""
        return {
            "enriched_at": datetime.utcnow().isoformat(),
            "processing_timestamp": datetime.utcnow().timestamp()
        }
    
    def _add_length_metrics(self, chunk: DocumentChunk, document: ProcessedDocument) -> Dict[str, Any]:
        """Add length-related metrics."""
        content_length = len(chunk.content)
        word_count = len(chunk.content.split())
        line_count = len(chunk.content.split('\n'))
        
        return {
            "content_length": content_length,
            "word_count": word_count,
            "line_count": line_count,
            "average_word_length": content_length / word_count if word_count > 0 else 0
        }
    
    def _add_semantic_metadata(self, chunk: DocumentChunk, document: ProcessedDocument) -> Dict[str, Any]:
        """Add semantic metadata based on content analysis."""
        semantic_metadata = {}
        
        # Detect language patterns
        semantic_metadata["has_numbers"] = any(char.isdigit() for char in chunk.content)
        semantic_metadata["has_special_chars"] = any(not char.isalnum() and not char.isspace() for char in chunk.content)
        
        # Detect content type indicators
        semantic_metadata["is_likely_code"] = self._is_likely_code(chunk.content)
        semantic_metadata["is_likely_table"] = self._is_likely_table(chunk.content)
        semantic_metadata["is_likely_list"] = self._is_likely_list(chunk.content)
        
        # Detect complexity indicators
        semantic_metadata["complexity_score"] = self._calculate_complexity_score(chunk.content)
        
        return semantic_metadata
    
    def _add_modality_specific_metadata(self, chunk: DocumentChunk, document: ProcessedDocument) -> Dict[str, Any]:
        """Add modality-specific metadata."""
        modality_metadata = {}
        
        if chunk.modality.value == "table":
            modality_metadata.update(self._extract_table_metadata(chunk.content))
        elif chunk.modality.value == "code":
            modality_metadata.update(self._extract_code_metadata(chunk.content))
        elif chunk.modality.value == "image":
            modality_metadata.update(self._extract_image_metadata(chunk.content))
        
        return modality_metadata
    
    def _is_likely_code(self, content: str) -> bool:
        """Detect if content is likely code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if __name__',
            'try:', 'except:', 'with ', 'for ', 'while ',
            'return ', 'print(', 'len(', 'range(',
        ]
        
        return any(indicator in content for indicator in code_indicators)
    
    def _is_likely_table(self, content: str) -> bool:
        """Detect if content is likely a table."""
        lines = content.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for consistent separators
        has_pipes = any('|' in line for line in lines)
        has_tabs = any('\t' in line for line in lines)
        
        return has_pipes or has_tabs
    
    def _is_likely_list(self, content: str) -> bool:
        """Detect if content is likely a list."""
        lines = content.split('\n')
        list_indicators = 0
        
        for line in lines:
            if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')):
                list_indicators += 1
        
        return list_indicators >= len(lines) * 0.5  # At least 50% of lines are list items
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate a simple complexity score."""
        if not content:
            return 0.0
        
        # Simple complexity metrics
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Normalize to 0-1 scale
        complexity = min(avg_sentence_length / 20.0, 1.0)  # Cap at 20 words per sentence
        
        return round(complexity, 3)
    
    def _extract_table_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata specific to table content."""
        lines = content.split('\n')
        
        return {
            "table_rows": len(lines),
            "has_header": len(lines) > 0 and any('|' in line or '\t' in line for line in lines[:1]),
            "separator_type": "pipe" if any('|' in line for line in lines) else "tab" if any('\t' in line for line in lines) else "space"
        }
    
    def _extract_code_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata specific to code content."""
        lines = content.split('\n')
        
        return {
            "code_lines": len(lines),
            "indentation_level": self._calculate_indentation_level(content),
            "has_comments": any(line.strip().startswith('#') for line in lines),
            "has_functions": any('def ' in line for line in lines),
            "has_classes": any('class ' in line for line in lines)
        }
    
    def _extract_image_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata specific to image content."""
        import re
        
        # Extract image file extensions
        image_extensions = re.findall(r'\.(jpg|jpeg|png|gif|bmp|tiff|svg)\b', content, re.IGNORECASE)
        
        return {
            "image_count": len(image_extensions),
            "image_types": list(set(image_extensions)),
            "has_caption": "caption" in content.lower() or "figure" in content.lower()
        }
    
    def _calculate_indentation_level(self, content: str) -> int:
        """Calculate the maximum indentation level in code."""
        lines = content.split('\n')
        max_indent = 0
        
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        return max_indent // 4  # Assume 4 spaces per indentation level 