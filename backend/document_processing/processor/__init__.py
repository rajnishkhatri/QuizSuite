"""
Processor Package

Document processing components for ingestion, chunking, cleaning, and enrichment.
"""

from .document_ingestor import DocumentIngestor, PDFDocumentIngestor, DocumentIngestionManager
from .chunking_strategy import ChunkingStrategy, ModalityAwareChunkingStrategy
from .metadata_enricher import MetadataEnricher
from .text_cleaner import (
    TextCleaner,
    StandardTextCleaner,
    ModalityAwareTextCleaner,
    CodeTextCleaner,
    TableTextCleaner,
    ImageTextCleaner,
    FigureTextCleaner
)

__all__ = [
    "DocumentIngestor",
    "PDFDocumentIngestor", 
    "DocumentIngestionManager",
    "ChunkingStrategy",
    "ModalityAwareChunkingStrategy",
    "MetadataEnricher",
    "TextCleaner",
    "StandardTextCleaner",
    "ModalityAwareTextCleaner",
    "CodeTextCleaner",
    "TableTextCleaner",
    "ImageTextCleaner",
    "FigureTextCleaner"
] 