"""
Text Cleaner Module

Cleans and normalizes text content from document chunks.
Follows Single Responsibility Principle - only responsible for text cleaning.
"""

import logging
import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from .models import DocumentChunk, ModalityType


logger = logging.getLogger(__name__)


class TextCleaner(ABC):
    """
    Abstract base class for text cleaning strategies.
    
    Follows Strategy Pattern for different cleaning approaches.
    """
    
    @abstractmethod
    def clean_text(self, content: str) -> str:
        """Clean text content according to this strategy."""
        pass


class StandardTextCleaner(TextCleaner):
    """
    Standard text cleaner that handles common text cleaning tasks.
    
    Follows Single Responsibility Principle - only handles text cleaning.
    """
    
    def __init__(self) -> None:
        """Initialize the standard text cleaner."""
        self.cleaning_patterns = self._initialize_cleaning_patterns()
    
    def _initialize_cleaning_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for text cleaning."""
        return {
            "multiple_spaces": r'\s+',
            "multiple_newlines": r'\n\s*\n\s*\n+',
            "leading_trailing_whitespace": r'^\s+|\s+$',
            "control_characters": r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
            "non_printable": r'[^\x20-\x7E\n\t]',
        }
    
    def clean_text(self, content: str) -> str:
        """Clean text content using standard cleaning patterns."""
        if not content:
            return content
        
        cleaned_content = content
        
        # Apply cleaning patterns
        cleaned_content = self._remove_control_characters(cleaned_content)
        cleaned_content = self._normalize_whitespace(cleaned_content)
        cleaned_content = self._fix_common_issues(cleaned_content)
        cleaned_content = self._trim_whitespace(cleaned_content)
        
        return cleaned_content
    
    def _remove_control_characters(self, content: str) -> str:
        """Remove control characters and non-printable characters."""
        # Remove control characters
        content = re.sub(self.cleaning_patterns["control_characters"], '', content)
        
        # Remove non-printable characters except newlines and tabs
        content = re.sub(self.cleaning_patterns["non_printable"], '', content)
        
        return content
    
    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace patterns."""
        # Replace multiple spaces with single space
        content = re.sub(self.cleaning_patterns["multiple_spaces"], ' ', content)
        
        # Replace multiple newlines with double newline
        content = re.sub(self.cleaning_patterns["multiple_newlines"], '\n\n', content)
        
        return content
    
    def _fix_common_issues(self, content: str) -> str:
        """Fix common text issues."""
        # Fix common OCR issues
        content = self._fix_ocr_issues(content)
        
        # Fix common formatting issues
        content = self._fix_formatting_issues(content)
        
        return content
    
    def _fix_ocr_issues(self, content: str) -> str:
        """Fix common OCR-related issues."""
        # Common OCR character replacements
        ocr_fixes = {
            '0': 'O',  # Common OCR mistake
            '1': 'l',  # Common OCR mistake
            '|': 'I',  # Common OCR mistake
        }
        
        # Only apply fixes in specific contexts to avoid over-correction
        for wrong_char, correct_char in ocr_fixes.items():
            # Only replace when it makes sense (e.g., in word contexts)
            content = re.sub(rf'\b{wrong_char}\b', correct_char, content)
        
        return content
    
    def _fix_formatting_issues(self, content: str) -> str:
        """Fix common formatting issues."""
        # Fix spacing around punctuation
        content = re.sub(r'\s+([.,;:!?])', r'\1', content)
        
        # Fix spacing around parentheses
        content = re.sub(r'\(\s+', '(', content)
        content = re.sub(r'\s+\)', ')', content)
        
        # Fix spacing around quotes
        content = re.sub(r'"\s+', '"', content)
        content = re.sub(r'\s+"', '"', content)
        
        return content
    
    def _trim_whitespace(self, content: str) -> str:
        """Remove leading and trailing whitespace."""
        # Remove leading/trailing whitespace from each line
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        
        # Remove empty lines at beginning and end
        while cleaned_lines and not cleaned_lines[0]:
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)


class ModalityAwareTextCleaner(TextCleaner):
    """
    Text cleaner that applies different cleaning strategies based on content modality.
    
    Follows Strategy Pattern and Single Responsibility Principle.
    """
    
    def __init__(self) -> None:
        """Initialize the modality-aware text cleaner."""
        self.standard_cleaner = StandardTextCleaner()
        self.modality_cleaners = self._initialize_modality_cleaners()
    
    def _initialize_modality_cleaners(self) -> Dict[ModalityType, TextCleaner]:
        """Initialize cleaners for different modalities."""
        return {
            ModalityType.TEXT: StandardTextCleaner(),
            ModalityType.CODE: CodeTextCleaner(),
            ModalityType.TABLE: TableTextCleaner(),
            ModalityType.IMAGE: ImageTextCleaner(),
            ModalityType.FIGURE: FigureTextCleaner(),
        }
    
    def clean_text(self, content: str) -> str:
        """Clean text content using modality-appropriate strategy."""
        # Default to standard cleaner
        return self.standard_cleaner.clean_text(content)
    
    def clean_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Clean a document chunk using modality-appropriate strategy."""
        cleaner = self.modality_cleaners.get(chunk.modality, self.standard_cleaner)
        cleaned_content = cleaner.clean_text(chunk.content)
        
        # Create new chunk with cleaned content
        return DocumentChunk(
            content=cleaned_content,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            modality=chunk.modality,
            metadata=chunk.metadata,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index
        )
    
    def clean_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Clean multiple chunks using appropriate strategies."""
        cleaned_chunks = []
        
        for chunk in chunks:
            try:
                cleaned_chunk = self.clean_chunk(chunk)
                cleaned_chunks.append(cleaned_chunk)
            except Exception as e:
                logger.error(f"Error cleaning chunk {chunk.chunk_id}: {e}")
                # Keep original chunk if cleaning fails
                cleaned_chunks.append(chunk)
        
        return cleaned_chunks


class CodeTextCleaner(TextCleaner):
    """Specialized cleaner for code content."""
    
    def clean_text(self, content: str) -> str:
        """Clean code content while preserving structure."""
        if not content:
            return content
        
        # Remove leading/trailing whitespace but preserve indentation
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve indentation but clean the rest
            stripped = line.strip()
            if stripped:
                # Find original indentation
                indent = len(line) - len(line.lstrip())
                cleaned_lines.append(' ' * indent + stripped)
        
        return '\n'.join(cleaned_lines)


class TableTextCleaner(TextCleaner):
    """Specialized cleaner for table content."""
    
    def clean_text(self, content: str) -> str:
        """Clean table content while preserving structure."""
        if not content:
            return content
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Clean table line while preserving separators
            cleaned_line = self._clean_table_line(line)
            if cleaned_line.strip():
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_table_line(self, line: str) -> str:
        """Clean a single table line."""
        # Normalize table separators
        line = re.sub(r'\s*\|\s*', ' | ', line)
        line = re.sub(r'\s*\t\s*', '\t', line)
        
        # Remove excessive whitespace
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()


class ImageTextCleaner(TextCleaner):
    """Specialized cleaner for image content."""
    
    def clean_text(self, content: str) -> str:
        """Clean image reference content."""
        if not content:
            return content
        
        # Clean image references while preserving file paths
        cleaned_content = content.strip()
        
        # Normalize image references
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        return cleaned_content


class FigureTextCleaner(TextCleaner):
    """Specialized cleaner for figure content."""
    
    def clean_text(self, content: str) -> str:
        """Clean figure reference content."""
        if not content:
            return content
        
        # Clean figure references
        cleaned_content = content.strip()
        
        # Normalize figure references
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
        
        return cleaned_content 