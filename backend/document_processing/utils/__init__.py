"""
Utils Package

Utility functions and helpers for document processing.
"""

from .reducers import first_value_reducer, dict_merge_reducer, list_merge_reducer
from .tracing import DocumentProcessingTracer, trace_document_processing_pipeline
from .error_handling import (
    DocumentProcessingError,
    IngestionError,
    ProcessingError,
    ChunkingError,
    CleaningError,
    EnrichmentError,
    ConfigurationError,
    ValidationError,
    ErrorHandler,
    error_handler,
    validate_file_path,
    validate_config,
    safe_execute,
    RetryHandler
)

__all__ = [
    "first_value_reducer",
    "dict_merge_reducer", 
    "list_merge_reducer",
    "DocumentProcessingTracer",
    "trace_document_processing_pipeline",
    "DocumentProcessingError",
    "IngestionError",
    "ProcessingError",
    "ChunkingError",
    "CleaningError",
    "EnrichmentError",
    "ConfigurationError",
    "ValidationError",
    "ErrorHandler",
    "error_handler",
    "validate_file_path",
    "validate_config",
    "safe_execute",
    "RetryHandler"
] 