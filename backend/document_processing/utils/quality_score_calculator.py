#!/usr/bin/env python3
"""
Quality Score Calculator

This module provides utilities to calculate quality scores for chunks.
Quality scores help prioritize chunks for processing and filtering.
"""

import re
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScoreResult:
    """Result of quality score calculation."""
    quality_score: float
    breakdown: Dict[str, float]
    factors: List[str]


class QualityScoreCalculator:
    """
    Calculates quality scores for chunks based on various criteria.
    
    Quality factors include:
    - Content length (optimal range)
    - Word count (optimal range)
    - Sentence count (minimum)
    - Meaningful content (domain-specific keywords)
    - Readability (sentence structure)
    - Information density (unique words ratio)
    - Modality-specific factors (OCR success for images)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the quality score calculator with configuration."""
        self.config = config or {}
        
        # Quality scoring settings
        scoring_config = self.config.get('quality_scoring', {})
        
        # Length scoring
        self.optimal_length_min = scoring_config.get('optimal_length_min', 200)
        self.optimal_length_max = scoring_config.get('optimal_length_max', 1000)
        self.min_length = scoring_config.get('min_length', 50)
        self.max_length = scoring_config.get('max_length', 5000)
        
        # Word count scoring
        self.optimal_words_min = scoring_config.get('optimal_words_min', 20)
        self.optimal_words_max = scoring_config.get('optimal_words_max', 200)
        self.min_words = scoring_config.get('min_words', 10)
        self.max_words = scoring_config.get('max_words', 1000)
        
        # Sentence scoring
        self.min_sentences = scoring_config.get('min_sentences', 1)
        self.optimal_sentences_min = scoring_config.get('optimal_sentences_min', 2)
        self.optimal_sentences_max = scoring_config.get('optimal_sentences_max', 10)
        
        # Content quality scoring
        self.meaningful_keywords = scoring_config.get('meaningful_keywords', [
            'architecture', 'enterprise', 'business', 'technology', 'system', 'process',
            'TOGAF', 'ADM', 'framework', 'principle', 'stakeholder', 'vision', 'driver',
            'requirement', 'component', 'service', 'integration', 'governance', 'strategy'
        ])
        
        # Modality-specific scoring
        self.ocr_success_weight = scoring_config.get('ocr_success_weight', 0.3)
        self.image_quality_weight = scoring_config.get('image_quality_weight', 0.2)
        
        logger.info(f"✅ Quality score calculator initialized")
    
    def calculate_quality_score(self, chunk: Dict[str, Any]) -> QualityScoreResult:
        """
        Calculate quality score for a chunk.
        
        Args:
            chunk: Chunk dictionary with content and metadata
            
        Returns:
            QualityScoreResult with score, breakdown, and factors
        """
        try:
            content = chunk.get('content', '')
            chunk_type = chunk.get('type', 'text')
            metadata = chunk.get('metadata', {})
            
            if not content:
                return QualityScoreResult(
                    quality_score=0.0,
                    breakdown={'content_length': 0.0, 'word_count': 0.0, 'sentences': 0.0, 'meaningful_content': 0.0},
                    factors=['no_content']
                )
            
            # Calculate individual quality factors
            length_score = self._calculate_length_score(content)
            word_score = self._calculate_word_score(content)
            sentence_score = self._calculate_sentence_score(content)
            meaningful_score = self._calculate_meaningful_content_score(content)
            
            # Calculate modality-specific scores
            modality_score = self._calculate_modality_score(chunk_type, metadata)
            
            # Calculate information density
            density_score = self._calculate_information_density_score(content)
            
            # Combine scores with weights
            breakdown = {
                'content_length': length_score,
                'word_count': word_score,
                'sentences': sentence_score,
                'meaningful_content': meaningful_score,
                'modality': modality_score,
                'information_density': density_score
            }
            
            # Weighted average (can be adjusted based on importance)
            weights = {
                'content_length': 0.2,
                'word_count': 0.2,
                'sentences': 0.15,
                'meaningful_content': 0.25,
                'modality': 0.1,
                'information_density': 0.1
            }
            
            total_score = sum(breakdown[factor] * weights[factor] for factor in breakdown)
            
            # Identify positive factors
            factors = []
            if length_score > 0.7:
                factors.append('optimal_length')
            if word_score > 0.7:
                factors.append('good_word_count')
            if sentence_score > 0.7:
                factors.append('good_sentence_structure')
            if meaningful_score > 0.7:
                factors.append('meaningful_content')
            if modality_score > 0.7:
                factors.append('good_modality')
            if density_score > 0.7:
                factors.append('high_information_density')
            
            return QualityScoreResult(
                quality_score=total_score,
                breakdown=breakdown,
                factors=factors
            )
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return QualityScoreResult(
                quality_score=0.0,
                breakdown={'error': 0.0},
                factors=['calculation_error']
            )
    
    def _calculate_length_score(self, content: str) -> float:
        """Calculate score based on content length."""
        length = len(content)
        
        if length < self.min_length:
            return 0.0
        elif length > self.max_length:
            return 0.3
        elif self.optimal_length_min <= length <= self.optimal_length_max:
            return 1.0
        else:
            # Linear interpolation for lengths between min/max and optimal
            if length < self.optimal_length_min:
                return 0.5 + (length - self.min_length) / (self.optimal_length_min - self.min_length) * 0.5
            else:
                return 0.5 + (self.max_length - length) / (self.max_length - self.optimal_length_max) * 0.5
    
    def _calculate_word_score(self, content: str) -> float:
        """Calculate score based on word count."""
        words = content.split()
        word_count = len(words)
        
        if word_count < self.min_words:
            return 0.0
        elif word_count > self.max_words:
            return 0.3
        elif self.optimal_words_min <= word_count <= self.optimal_words_max:
            return 1.0
        else:
            # Linear interpolation
            if word_count < self.optimal_words_min:
                return 0.5 + (word_count - self.min_words) / (self.optimal_words_min - self.min_words) * 0.5
            else:
                return 0.5 + (self.max_words - word_count) / (self.max_words - self.optimal_words_max) * 0.5
    
    def _calculate_sentence_score(self, content: str) -> float:
        """Calculate score based on sentence count and structure."""
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count < self.min_sentences:
            return 0.0
        elif self.optimal_sentences_min <= sentence_count <= self.optimal_sentences_max:
            return 1.0
        else:
            # Penalize too few or too many sentences
            if sentence_count < self.optimal_sentences_min:
                return sentence_count / self.optimal_sentences_min
            else:
                return max(0.3, 1.0 - (sentence_count - self.optimal_sentences_max) / 10)
    
    def _calculate_meaningful_content_score(self, content: str) -> float:
        """Calculate score based on presence of meaningful keywords."""
        content_lower = content.lower()
        found_keywords = []
        
        for keyword in self.meaningful_keywords:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        # Score based on keyword density and variety
        keyword_density = len(found_keywords) / len(self.meaningful_keywords)
        unique_keywords = len(set(found_keywords))
        
        # Combine density and variety
        density_score = min(1.0, keyword_density * 2)  # Scale up density
        variety_score = min(1.0, unique_keywords / 5)  # Bonus for variety
        
        return (density_score + variety_score) / 2
    
    def _calculate_modality_score(self, chunk_type: str, metadata: Dict[str, Any]) -> float:
        """Calculate modality-specific quality score."""
        if chunk_type == 'image':
            # For images, consider OCR success and image quality
            ocr_success = metadata.get('ocr_success', False)
            image_quality = self._assess_image_quality(metadata)
            
            ocr_score = 1.0 if ocr_success else 0.3
            quality_score = image_quality
            
            return (ocr_score * self.ocr_success_weight + quality_score * self.image_quality_weight) / (self.ocr_success_weight + self.image_quality_weight)
        
        elif chunk_type == 'table':
            # For tables, consider structure and data density
            rows = metadata.get('rows', 0)
            columns = metadata.get('columns', 0)
            non_empty_cells = metadata.get('non_empty_cells', 0)
            total_cells = metadata.get('total_cells', 0)
            
            if total_cells == 0:
                return 0.0
            
            structure_score = min(1.0, (rows + columns) / 10)  # Bonus for larger tables
            data_density = non_empty_cells / total_cells if total_cells > 0 else 0.0
            
            return (structure_score + data_density) / 2
        
        elif chunk_type == 'text':
            # Text chunks get base score
            return 0.8
        
        else:
            # Default score for other modalities
            return 0.6
    
    def _assess_image_quality(self, metadata: Dict[str, Any]) -> float:
        """Assess image quality based on metadata."""
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        size_bytes = metadata.get('size_bytes', 0)
        
        # Simple quality assessment based on dimensions and size
        if width == 0 or height == 0:
            return 0.3
        
        # Larger images generally have better quality
        area = width * height
        if area > 1000000:  # 1M pixels
            return 1.0
        elif area > 100000:  # 100K pixels
            return 0.8
        elif area > 10000:  # 10K pixels
            return 0.6
        else:
            return 0.4
    
    def _calculate_information_density_score(self, content: str) -> float:
        """Calculate information density score."""
        words = content.split()
        if not words:
            return 0.0
        
        # Calculate unique word ratio
        unique_words = len(set(words))
        total_words = len(words)
        unique_ratio = unique_words / total_words if total_words > 0 else 0.0
        
        # Penalize repetitive content
        if unique_ratio < 0.3:
            return 0.2
        elif unique_ratio < 0.5:
            return 0.5
        elif unique_ratio < 0.7:
            return 0.8
        else:
            return 1.0
    
    def calculate_chunks_quality_scores(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate quality scores for multiple chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with quality scores added
        """
        scored_chunks = []
        
        for chunk in chunks:
            quality_result = self.calculate_quality_score(chunk)
            
            # Add quality score to chunk
            chunk_with_score = chunk.copy()
            chunk_with_score['quality_score'] = quality_result.quality_score
            chunk_with_score['quality_breakdown'] = quality_result.breakdown
            chunk_with_score['quality_factors'] = quality_result.factors
            
            scored_chunks.append(chunk_with_score)
        
        return scored_chunks


def create_quality_score_calculator(config: Dict[str, Any] = None) -> QualityScoreCalculator:
    """Create a quality score calculator instance."""
    return QualityScoreCalculator(config)


def calculate_chunks_quality_scores(chunks: List[Dict[str, Any]], config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Calculate quality scores for chunks."""
    calculator = create_quality_score_calculator(config)
    return calculator.calculate_chunks_quality_scores(chunks) 