"""
Metadata Enricher

Component for enriching document chunks with additional metadata.
"""

from typing import List, Dict, Any
from datetime import datetime

from ..model.document_models import DocumentChunk


class MetadataEnricher:
    """Enriches document chunks with additional metadata."""
    
    def __init__(self):
        """Initialize the metadata enricher."""
        pass
    
    def enrich_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Enrich a list of document chunks with additional metadata.
        
        Args:
            chunks: List of document chunks to enrich
            
        Returns:
            List of enriched document chunks
        """
        enriched_chunks = []
        
        for chunk in chunks:
            enriched_chunk = self._enrich_single_chunk(chunk)
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _enrich_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Enrich a single document chunk with metadata.
        
        Args:
            chunk: The chunk to enrich
            
        Returns:
            Enriched document chunk
        """
        # Create a copy of the chunk with enriched metadata
        enriched_metadata = chunk.metadata.copy()
        
        # Add basic metadata
        enriched_metadata.update({
            "enrichment_timestamp": datetime.utcnow().isoformat(),
            "content_length": len(chunk.content),
            "word_count": len(chunk.content.split()),
            "enriched": True
        })
        
        # Add modality-specific metadata
        if chunk.modality.value == "text":
            enriched_metadata.update(self._enrich_text_metadata(chunk))
        elif chunk.modality.value == "table":
            enriched_metadata.update(self._enrich_table_metadata(chunk))
        elif chunk.modality.value == "image":
            enriched_metadata.update(self._enrich_image_metadata(chunk))
        elif chunk.modality.value == "figure":
            enriched_metadata.update(self._enrich_figure_metadata(chunk))
        elif chunk.modality.value == "code":
            enriched_metadata.update(self._enrich_code_metadata(chunk))
        
        # Create new chunk with enriched metadata
        enriched_chunk = DocumentChunk(
            content=chunk.content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=enriched_metadata,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding
        )
        
        return enriched_chunk
    
    def _enrich_text_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Enrich text chunk with text-specific metadata.
        
        Args:
            chunk: The text chunk to enrich
            
        Returns:
            Dictionary of text-specific metadata
        """
        content = chunk.content
        
        return {
            "text_metadata": {
                "sentence_count": len([s for s in content.split('.') if s.strip()]),
                "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
                "has_numbers": any(char.isdigit() for char in content),
                "has_special_chars": any(not char.isalnum() and char != ' ' for char in content),
                "avg_word_length": sum(len(word) for word in content.split()) / max(len(content.split()), 1)
            }
        }
    
    def _enrich_table_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Enrich table chunk with table-specific metadata.
        
        Args:
            chunk: The table chunk to enrich
            
        Returns:
            Dictionary of table-specific metadata
        """
        content = chunk.content
        
        return {
            "table_metadata": {
                "row_count": len([line for line in content.split('\n') if '|' in line]),
                "column_count": max(len(line.split('|')) for line in content.split('\n') if '|' in line) if any('|' in line for line in content.split('\n')) else 0,
                "has_header": any('|' in line and any(word.isupper() for word in line.split('|')) for line in content.split('\n')),
                "has_numeric_data": any(char.isdigit() for char in content)
            }
        }
    
    def _enrich_image_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Enrich image chunk with image-specific metadata.
        
        Args:
            chunk: The image chunk to enrich
            
        Returns:
            Dictionary of image-specific metadata
        """
        return {
            "image_metadata": {
                "content_type": "image",
                "has_caption": "caption" in chunk.content.lower() or "figure" in chunk.content.lower(),
                "caption_length": len(chunk.content)
            }
        }
    
    def _enrich_figure_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Enrich figure chunk with figure-specific metadata.
        
        Args:
            chunk: The figure chunk to enrich
            
        Returns:
            Dictionary of figure-specific metadata
        """
        return {
            "figure_metadata": {
                "content_type": "figure",
                "has_caption": "caption" in chunk.content.lower() or "figure" in chunk.content.lower(),
                "caption_length": len(chunk.content),
                "figure_type": self._detect_figure_type(chunk.content)
            }
        }
    
    def _enrich_code_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Enrich code chunk with code-specific metadata.
        
        Args:
            chunk: The code chunk to enrich
            
        Returns:
            Dictionary of code-specific metadata
        """
        content = chunk.content
        
        return {
            "code_metadata": {
                "line_count": len(content.split('\n')),
                "has_comments": any(line.strip().startswith('#') for line in content.split('\n')),
                "has_functions": any('def ' in line for line in content.split('\n')),
                "has_classes": any('class ' in line for line in content.split('\n')),
                "estimated_language": self._detect_code_language(content)
            }
        }
    
    def _detect_figure_type(self, content: str) -> str:
        """Detect the type of figure based on content.
        
        Args:
            content: The figure content
            
        Returns:
            Detected figure type
        """
        content_lower = content.lower()
        
        if "chart" in content_lower or "graph" in content_lower:
            return "chart"
        elif "diagram" in content_lower:
            return "diagram"
        elif "photo" in content_lower or "image" in content_lower:
            return "image"
        else:
            return "unknown"
    
    def _detect_code_language(self, content: str) -> str:
        """Detect the programming language of code content.
        
        Args:
            content: The code content
            
        Returns:
            Detected programming language
        """
        content_lower = content.lower()
        
        if "def " in content_lower and "import " in content_lower:
            return "python"
        elif "function " in content_lower and "var " in content_lower:
            return "javascript"
        elif "public class" in content_lower or "private " in content_lower:
            return "java"
        elif "<?php" in content_lower:
            return "php"
        elif "def " in content_lower:
            return "python"
        else:
            return "unknown" 