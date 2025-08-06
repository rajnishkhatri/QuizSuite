#!/usr/bin/env python3
"""
Chunk Content Validator

This module provides utilities to validate chunk content before sending to LLM.
Invalid chunks are filtered out to prevent timeouts and improve efficiency.
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkValidationResult:
    """Result of chunk validation."""
    is_valid: bool
    reason: str
    cleaned_content: str = ""
    validation_score: float = 0.0


class ChunkContentValidator:
    """
    Validates chunk content before sending to LLM.
    
    Filters out chunks that are:
    - Too short or too long
    - Empty or whitespace-only
    - Contains only URLs, numbers, or special characters
    - Contains repetitive content
    - Contains non-English content (optional)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the validator with configuration."""
        self.config = config or {}
        
        # Default validation settings
        self.min_length = self.config.get("min_length", 50)
        self.max_length = self.config.get("max_length", 10000)
        self.min_words = self.config.get("min_words", 10)
        self.max_words = self.config.get("max_words", 2000)
        self.min_sentences = self.config.get("min_sentences", 1)
        self.max_repetition_ratio = self.config.get("max_repetition_ratio", 0.3)
        self.require_meaningful_content = self.config.get("require_meaningful_content", True)
        self.exclude_patterns = self.config.get("exclude_patterns", [
            r'^\s*$',  # Empty or whitespace-only
            r'^[0-9\s\-\.]+$',  # Only numbers and punctuation
            r'^[A-Z\s]+$',  # Only uppercase letters
            r'^https?://',  # Only URLs
            r'^[^\w\s]+$',  # Only special characters
        ])
        self.include_patterns = self.config.get("include_patterns", [
            r'\b(architecture|enterprise|business|technology|system|process|data|application)\b',
            r'\b(TOGAF|ADM|framework|principle|stakeholder|vision|driver|requirement)\b',
            r'\b(component|service|integration|governance|strategy|implementation)\b',
        ])
        
        # Compile regex patterns
        self.exclude_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.exclude_patterns]
        self.include_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.include_patterns]
    
    def validate_chunk(self, chunk: Dict[str, Any]) -> ChunkValidationResult:
        """
        Validate a single chunk.
        
        Args:
            chunk: Chunk dictionary with content and metadata
            
        Returns:
            ChunkValidationResult with validation status and details
        """
        try:
            # Extract content
            content = self._extract_content(chunk)
            if not content:
                return ChunkValidationResult(
                    is_valid=False,
                    reason="No content found in chunk",
                    validation_score=0.0
                )
            
            # Clean content
            cleaned_content = self._clean_content(content)
            
            # Perform validations
            validation_results = []
            
            # 1. Length validation
            length_result = self._validate_length(cleaned_content)
            validation_results.append(length_result)
            
            # 2. Word count validation
            word_result = self._validate_word_count(cleaned_content)
            validation_results.append(word_result)
            
            # 3. Sentence count validation
            sentence_result = self._validate_sentence_count(cleaned_content)
            validation_results.append(sentence_result)
            
            # 4. Pattern validation
            pattern_result = self._validate_patterns(cleaned_content)
            validation_results.append(pattern_result)
            
            # 5. Repetition validation
            repetition_result = self._validate_repetition(cleaned_content)
            validation_results.append(repetition_result)
            
            # 6. Meaningful content validation
            meaningful_result = self._validate_meaningful_content(cleaned_content)
            validation_results.append(meaningful_result)
            
            # Calculate overall validation score
            valid_results = [r for r in validation_results if r.is_valid]
            validation_score = len(valid_results) / len(validation_results) if validation_results else 0.0
            
            # Determine if chunk is valid
            is_valid = validation_score >= 0.8  # At least 80% of validations must pass
            
            # Get primary reason for invalidation
            if not is_valid:
                failed_results = [r for r in validation_results if not r.is_valid]
                primary_reason = failed_results[0].reason if failed_results else "Unknown validation failure"
            else:
                primary_reason = "All validations passed"
            
            logger.debug(f"Chunk validation: {is_valid}, score: {validation_score:.2f}, reason: {primary_reason}")
            
            return ChunkValidationResult(
                is_valid=is_valid,
                reason=primary_reason,
                cleaned_content=cleaned_content,
                validation_score=validation_score
            )
            
        except Exception as e:
            logger.error(f"Error validating chunk: {e}")
            return ChunkValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                validation_score=0.0
            )
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate multiple chunks and return valid and invalid chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (valid_chunks, invalid_chunks)
        """
        valid_chunks = []
        invalid_chunks = []
        
        for chunk in chunks:
            result = self.validate_chunk(chunk)
            
            if result.is_valid:
                # Add validation metadata to valid chunk
                chunk['validation_metadata'] = {
                    'validation_score': result.validation_score,
                    'cleaned_content': result.cleaned_content,
                    'validation_reason': result.reason
                }
                valid_chunks.append(chunk)
            else:
                # Add validation metadata to invalid chunk
                chunk['validation_metadata'] = {
                    'validation_score': result.validation_score,
                    'validation_reason': result.reason,
                    'is_invalid': True
                }
                invalid_chunks.append(chunk)
        
        logger.info(f"Chunk validation complete: {len(valid_chunks)} valid, {len(invalid_chunks)} invalid")
        
        return valid_chunks, invalid_chunks
    
    def _extract_content(self, chunk: Dict[str, Any]) -> str:
        """Extract content from chunk based on chunk type."""
        content = ""
        
        # Try different content fields based on chunk type
        if 'content' in chunk:
            content = chunk['content']
        elif 'text' in chunk:
            content = chunk['text']
        elif 'extracted_content' in chunk:
            extracted = chunk['extracted_content']
            if isinstance(extracted, dict):
                content = extracted.get('text', '')
            else:
                content = str(extracted)
        elif 'metadata' in chunk and 'content' in chunk['metadata']:
            content = chunk['metadata']['content']
        
        return str(content) if content else ""
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', content.strip())
        
        # Remove common noise patterns
        cleaned = re.sub(r'^\s*[•\-\*]\s*', '', cleaned)  # Remove bullet points
        cleaned = re.sub(r'\s*[•\-\*]\s*$', '', cleaned)  # Remove trailing bullets
        
        return cleaned
    
    def _validate_length(self, content: str) -> ChunkValidationResult:
        """Validate content length."""
        length = len(content)
        
        if length < self.min_length:
            return ChunkValidationResult(
                is_valid=False,
                reason=f"Content too short: {length} chars (min: {self.min_length})",
                validation_score=0.0
            )
        
        if length > self.max_length:
            return ChunkValidationResult(
                is_valid=False,
                reason=f"Content too long: {length} chars (max: {self.max_length})",
                validation_score=0.0
            )
        
        return ChunkValidationResult(
            is_valid=True,
            reason="Length validation passed",
            validation_score=1.0
        )
    
    def _validate_word_count(self, content: str) -> ChunkValidationResult:
        """Validate word count."""
        words = content.split()
        word_count = len(words)
        
        if word_count < self.min_words:
            return ChunkValidationResult(
                is_valid=False,
                reason=f"Too few words: {word_count} (min: {self.min_words})",
                validation_score=0.0
            )
        
        if word_count > self.max_words:
            return ChunkValidationResult(
                is_valid=False,
                reason=f"Too many words: {word_count} (max: {self.max_words})",
                validation_score=0.0
            )
        
        return ChunkValidationResult(
            is_valid=True,
            reason="Word count validation passed",
            validation_score=1.0
        )
    
    def _validate_sentence_count(self, content: str) -> ChunkValidationResult:
        """Validate sentence count."""
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count < self.min_sentences:
            return ChunkValidationResult(
                is_valid=False,
                reason=f"Too few sentences: {sentence_count} (min: {self.min_sentences})",
                validation_score=0.0
            )
        
        return ChunkValidationResult(
            is_valid=True,
            reason="Sentence count validation passed",
            validation_score=1.0
        )
    
    def _validate_patterns(self, content: str) -> ChunkValidationResult:
        """Validate content against include/exclude patterns."""
        # Check exclude patterns
        for pattern in self.exclude_regex:
            if pattern.search(content):
                return ChunkValidationResult(
                    is_valid=False,
                    reason=f"Content matches exclude pattern: {pattern.pattern}",
                    validation_score=0.0
                )
        
        # Check include patterns (at least one should match)
        if self.include_patterns:
            has_include_match = any(pattern.search(content) for pattern in self.include_regex)
            if not has_include_match:
                return ChunkValidationResult(
                    is_valid=False,
                    reason="Content doesn't match any include patterns",
                    validation_score=0.0
                )
        
        return ChunkValidationResult(
            is_valid=True,
            reason="Pattern validation passed",
            validation_score=1.0
        )
    
    def _validate_repetition(self, content: str) -> ChunkValidationResult:
        """Validate content for excessive repetition."""
        words = content.lower().split()
        if len(words) < 5:
            return ChunkValidationResult(
                is_valid=True,
                reason="Too few words for repetition check",
                validation_score=1.0
            )
        
        # Calculate repetition ratio
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Ignore very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            max_freq = max(word_freq.values())
            repetition_ratio = max_freq / len(words)
            
            if repetition_ratio > self.max_repetition_ratio:
                return ChunkValidationResult(
                    is_valid=False,
                    reason=f"Excessive repetition: {repetition_ratio:.2f} (max: {self.max_repetition_ratio})",
                    validation_score=0.0
                )
        
        return ChunkValidationResult(
            is_valid=True,
            reason="Repetition validation passed",
            validation_score=1.0
        )
    
    def _validate_meaningful_content(self, content: str) -> ChunkValidationResult:
        """Validate that content has meaningful information."""
        if not self.require_meaningful_content:
            return ChunkValidationResult(
                is_valid=True,
                reason="Meaningful content validation skipped",
                validation_score=1.0
            )
        
        # Check for meaningful content indicators
        meaningful_indicators = [
            r'\b(architecture|enterprise|business|technology|system|process|data|application)\b',
            r'\b(TOGAF|ADM|framework|principle|stakeholder|vision|driver|requirement)\b',
            r'\b(component|service|integration|governance|strategy|implementation)\b',
            r'\b(design|planning|development|management|analysis|assessment)\b',
        ]
        
        meaningful_count = 0
        for pattern in meaningful_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                meaningful_count += 1
        
        if meaningful_count == 0:
            return ChunkValidationResult(
                is_valid=False,
                reason="No meaningful content indicators found",
                validation_score=0.0
            )
        
        return ChunkValidationResult(
            is_valid=True,
            reason=f"Meaningful content found: {meaningful_count} indicators",
            validation_score=1.0
        )


def create_chunk_validator(config: Dict[str, Any] = None) -> ChunkContentValidator:
    """Factory function to create a chunk validator."""
    return ChunkContentValidator(config)


def validate_chunks_for_llm(chunks: List[Dict[str, Any]], config: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience function to validate chunks before sending to LLM.
    
    Args:
        chunks: List of chunks to validate
        config: Optional validation configuration
        
    Returns:
        Tuple of (valid_chunks, invalid_chunks)
    """
    validator = create_chunk_validator(config)
    return validator.validate_chunks(chunks) 