"""
Error Handling Utilities

Provides comprehensive error handling for the document processing pipeline.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable, Type
from functools import wraps
from pathlib import Path

from .tracing import DocumentProcessingTracer

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Initialize the error.
        
        Args:
            message: Error message
            context: Additional context about the error
        """
        super().__init__(message)
        self.context = context or {}
        self.timestamp = None  # Will be set by error handler


class IngestionError(DocumentProcessingError):
    """Error during document ingestion."""
    pass


class ProcessingError(DocumentProcessingError):
    """Error during document processing."""
    pass


class ChunkingError(DocumentProcessingError):
    """Error during document chunking."""
    pass


class CleaningError(DocumentProcessingError):
    """Error during text cleaning."""
    pass


class EnrichmentError(DocumentProcessingError):
    """Error during metadata enrichment."""
    pass


class ConfigurationError(DocumentProcessingError):
    """Error with configuration."""
    pass


class ValidationError(DocumentProcessingError):
    """Error with data validation."""
    pass


class ErrorHandler:
    """Comprehensive error handler for document processing."""
    
    def __init__(self, tracer: Optional[DocumentProcessingTracer] = None):
        """Initialize the error handler.
        
        Args:
            tracer: Optional tracer for error tracking
        """
        self.tracer = tracer or DocumentProcessingTracer()
        self.error_counts = {
            "ingestion": 0,
            "processing": 0,
            "chunking": 0,
            "cleaning": 0,
            "enrichment": 0,
            "configuration": 0,
            "validation": 0,
            "other": 0
        }
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        error_type: str = "other",
        reraise: bool = True
    ) -> None:
        """Handle an error with comprehensive logging and tracing.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            error_type: Type of error for categorization
            reraise: Whether to reraise the exception
        """
        # Update error counts
        if error_type in self.error_counts:
            self.error_counts[error_type] += 1
        else:
            self.error_counts["other"] += 1
        
        # Log the error
        logger.error(
            f"Error in {error_type}: {error}",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "traceback": traceback.format_exc()
            }
        )
        
        # Trace the error
        try:
            self.tracer.trace_error(error, context)
        except Exception as trace_error:
            logger.warning(f"Failed to trace error: {trace_error}")
        
        # Reraise if requested
        if reraise:
            raise error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors encountered.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = sum(self.error_counts.values())
        return {
            "total_errors": total_errors,
            "error_breakdown": self.error_counts.copy(),
            "has_errors": total_errors > 0
        }
    
    def reset_error_counts(self) -> None:
        """Reset error counts."""
        for key in self.error_counts:
            self.error_counts[key] = 0


def error_handler(
    error_type: str = "other",
    reraise: bool = True,
    context_provider: Optional[Callable] = None
):
    """Decorator for error handling.
    
    Args:
        error_type: Type of error for categorization
        reraise: Whether to reraise the exception
        context_provider: Function to provide additional context
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            context = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            
            if context_provider:
                try:
                    additional_context = context_provider(*args, **kwargs)
                    context.update(additional_context)
                except Exception as e:
                    logger.warning(f"Context provider failed: {e}")
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.handle_error(e, context, error_type, reraise)
        
        return wrapper
    return decorator


def validate_file_path(file_path: Path) -> None:
    """Validate that a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Raises:
        ValidationError: If the file path is invalid
    """
    if not file_path.exists():
        raise ValidationError(f"File does not exist: {file_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")
    
    if not file_path.stat().st_size > 0:
        raise ValidationError(f"File is empty: {file_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    required_fields = ["model_name", "database_type", "embedding_model"]
    
    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required field: {field}")
        
        if not config[field]:
            raise ConfigurationError(f"Field {field} cannot be empty")


def safe_execute(
    func: Callable,
    *args,
    error_type: str = "other",
    context: Optional[Dict[str, Any]] = None,
    default_return: Any = None,
    **kwargs
) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        error_type: Type of error for categorization
        context: Additional context for error handling
        default_return: Value to return if function fails
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of function execution or default_return
    """
    handler = ErrorHandler()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_context = context or {}
        error_context.update({
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs)
        })
        
        handler.handle_error(e, error_context, error_type, reraise=False)
        return default_return


class RetryHandler:
    """Handler for retrying failed operations."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """Initialize the retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Factor for exponential backoff
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def retry(
        self,
        func: Callable,
        *args,
        error_types: Optional[tuple] = None,
        **kwargs
    ) -> Any:
        """Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Arguments for the function
            error_types: Tuple of exception types to retry on
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of function execution
            
        Raises:
            Last exception if all retries fail
        """
        if error_types is None:
            error_types = (Exception,)
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    
                    import time
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed for {func.__name__}: {e}"
                    )
        
        raise last_exception 