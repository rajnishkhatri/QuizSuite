"""
Text Cleaner

Abstract base class and concrete implementations for text cleaning strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import re

from ..model.document_models import DocumentChunk, ModalityType


class TextCleaner(ABC):
    """Abstract base class for text cleaning strategies."""
    
    def __init__(self):
        """Initialize text cleaner."""
        pass
    
    @abstractmethod
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean a list of document chunks.
        
        Args:
            chunks: List of document chunks to clean
            
        Returns:
            List of cleaned document chunks
        """
        pass


class StandardTextCleaner(TextCleaner):
    """Standard text cleaning implementation."""
    
    def __init__(self):
        """Initialize standard text cleaner."""
        super().__init__()
    
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean chunks using standard text cleaning.
        
        Args:
            chunks: List of document chunks to clean
            
        Returns:
            List of cleaned document chunks
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            cleaned_chunk = self._clean_single_chunk(chunk)
            cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks
    
    def _clean_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Clean a single document chunk.
        
        Args:
            chunk: The chunk to clean
            
        Returns:
            Cleaned document chunk
        """
        cleaned_content = self._clean_text(chunk.content)
        
        # Create new chunk with cleaned content
        cleaned_chunk = DocumentChunk(
            content=cleaned_content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=chunk.metadata.copy(),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding
        )
        
        # Add cleaning metadata
        cleaned_chunk.metadata.update({
            "cleaned": True,
            "original_length": len(chunk.content),
            "cleaned_length": len(cleaned_content),
            "cleaning_strategy": "standard"
        })
        
        return cleaned_chunk
    
    def _clean_text(self, text: str) -> str:
        """Clean text using standard cleaning rules.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text


class ModalityAwareTextCleaner(TextCleaner):
    """Text cleaner that is aware of different content modalities."""
    
    def __init__(self):
        """Initialize modality-aware text cleaner."""
        super().__init__()
        self.cleaners = {
            ModalityType.TEXT: StandardTextCleaner(),
            ModalityType.CODE: CodeTextCleaner(),
            ModalityType.TABLE: TableTextCleaner(),
            ModalityType.IMAGE: ImageTextCleaner(),
            ModalityType.FIGURE: FigureTextCleaner()
        }
    
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean chunks using modality-aware cleaning.
        
        Args:
            chunks: List of document chunks to clean
            
        Returns:
            List of cleaned document chunks
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            # Use appropriate cleaner based on modality
            cleaner = self.cleaners.get(chunk.modality, StandardTextCleaner())
            cleaned_chunk = cleaner.clean_chunks([chunk])[0]
            cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks


class CodeTextCleaner(TextCleaner):
    """Text cleaner specialized for code content."""
    
    def __init__(self):
        """Initialize code text cleaner."""
        super().__init__()
    
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean code chunks.
        
        Args:
            chunks: List of code chunks to clean
            
        Returns:
            List of cleaned code chunks
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            cleaned_chunk = self._clean_single_chunk(chunk)
            cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks
    
    def _clean_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Clean a single code chunk.
        
        Args:
            chunk: The code chunk to clean
            
        Returns:
            Cleaned code chunk
        """
        cleaned_content = self._clean_code(chunk.content)
        
        # Create new chunk with cleaned content
        cleaned_chunk = DocumentChunk(
            content=cleaned_content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=chunk.metadata.copy(),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding
        )
        
        # Add cleaning metadata
        cleaned_chunk.metadata.update({
            "cleaned": True,
            "cleaning_strategy": "code",
            "original_length": len(chunk.content),
            "cleaned_length": len(cleaned_content)
        })
        
        return cleaned_chunk
    
    def _clean_code(self, code: str) -> str:
        """Clean code text.
        
        Args:
            code: The code to clean
            
        Returns:
            Cleaned code
        """
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in code.split('\n')]
        
        # Remove empty lines at the beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Normalize indentation (preserve structure)
        return '\n'.join(lines)


class TableTextCleaner(TextCleaner):
    """Text cleaner specialized for table content."""
    
    def __init__(self):
        """Initialize table text cleaner."""
        super().__init__()
    
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean table chunks.
        
        Args:
            chunks: List of table chunks to clean
            
        Returns:
            List of cleaned table chunks
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            cleaned_chunk = self._clean_single_chunk(chunk)
            cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks
    
    def _clean_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Clean a single table chunk.
        
        Args:
            chunk: The table chunk to clean
            
        Returns:
            Cleaned table chunk
        """
        cleaned_content = self._clean_table(chunk.content)
        
        # Create new chunk with cleaned content
        cleaned_chunk = DocumentChunk(
            content=cleaned_content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=chunk.metadata.copy(),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding
        )
        
        # Add cleaning metadata
        cleaned_chunk.metadata.update({
            "cleaned": True,
            "cleaning_strategy": "table",
            "original_length": len(chunk.content),
            "cleaned_length": len(cleaned_content)
        })
        
        return cleaned_chunk
    
    def _clean_table(self, table: str) -> str:
        """Clean table text.
        
        Args:
            table: The table to clean
            
        Returns:
            Cleaned table
        """
        lines = table.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Clean table row
            cleaned_line = self._clean_table_row(line)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_table_row(self, row: str) -> str:
        """Clean a single table row.
        
        Args:
            row: The table row to clean
            
        Returns:
            Cleaned table row
        """
        # Remove extra whitespace around pipe separators
        row = re.sub(r'\s*\|\s*', '|', row)
        
        # Remove leading/trailing pipes if they're empty
        row = re.sub(r'^\|+', '', row)
        row = re.sub(r'\|+$', '', row)
        
        return row.strip()


class ImageTextCleaner(TextCleaner):
    """Text cleaner specialized for image content (captions, descriptions)."""
    
    def __init__(self):
        """Initialize image text cleaner."""
        super().__init__()
    
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean image chunks.
        
        Args:
            chunks: List of image chunks to clean
            
        Returns:
            List of cleaned image chunks
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            cleaned_chunk = self._clean_single_chunk(chunk)
            cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks
    
    def _clean_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Clean a single image chunk.
        
        Args:
            chunk: The image chunk to clean
            
        Returns:
            Cleaned image chunk
        """
        cleaned_content = self._clean_image_text(chunk.content)
        
        # Create new chunk with cleaned content
        cleaned_chunk = DocumentChunk(
            content=cleaned_content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=chunk.metadata.copy(),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding
        )
        
        # Add cleaning metadata
        cleaned_chunk.metadata.update({
            "cleaned": True,
            "cleaning_strategy": "image",
            "original_length": len(chunk.content),
            "cleaned_length": len(cleaned_content)
        })
        
        return cleaned_chunk
    
    def _clean_image_text(self, text: str) -> str:
        """Clean image text (captions, descriptions).
        
        Args:
            text: The image text to clean
            
        Returns:
            Cleaned image text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common image prefixes
        text = re.sub(r'^(Figure|Image|Photo|Picture)\s*\d*[:\s]*', '', text, flags=re.IGNORECASE)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text


class FigureTextCleaner(TextCleaner):
    """Text cleaner specialized for figure content."""
    
    def __init__(self):
        """Initialize figure text cleaner."""
        super().__init__()
    
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean figure chunks.
        
        Args:
            chunks: List of figure chunks to clean
            
        Returns:
            List of cleaned figure chunks
        """
        cleaned_chunks = []
        
        for chunk in chunks:
            cleaned_chunk = self._clean_single_chunk(chunk)
            cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks
    
    def _clean_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Clean a single figure chunk.
        
        Args:
            chunk: The figure chunk to clean
            
        Returns:
            Cleaned figure chunk
        """
        cleaned_content = self._clean_figure_text(chunk.content)
        
        # Create new chunk with cleaned content
        cleaned_chunk = DocumentChunk(
            content=cleaned_content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=chunk.metadata.copy(),
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding
        )
        
        # Add cleaning metadata
        cleaned_chunk.metadata.update({
            "cleaned": True,
            "cleaning_strategy": "figure",
            "original_length": len(chunk.content),
            "cleaned_length": len(cleaned_content)
        })
        
        return cleaned_chunk
    
    def _clean_figure_text(self, text: str) -> str:
        """Clean figure text.
        
        Args:
            text: The figure text to clean
            
        Returns:
            Cleaned figure text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common figure prefixes
        text = re.sub(r'^(Figure|Fig)\s*\d*[:\s]*', '', text, flags=re.IGNORECASE)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text 