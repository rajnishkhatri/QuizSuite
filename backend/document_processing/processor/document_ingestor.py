"""
Document Ingestor Module

Handles the ingestion of documents from various sources and formats.
Follows Single Responsibility Principle - only responsible for document ingestion.
"""

import logging
from pathlib import Path
from typing import List, Optional
from abc import ABC, abstractmethod

from ..model.document_models import DocumentType, ProcessedDocument


logger = logging.getLogger(__name__)


class DocumentIngestor(ABC):
    """
    Abstract base class for document ingestion.
    
    Follows Open/Closed Principle - open for extension, closed for modification.
    """
    
    def __init__(self) -> None:
        """Initialize the document ingestor."""
        self.supported_types = self._get_supported_types()
    
    @abstractmethod
    def _get_supported_types(self) -> List[DocumentType]:
        """Return list of supported document types."""
        pass
    
    @abstractmethod
    def can_ingest(self, file_path: Path) -> bool:
        """Check if this ingestor can handle the given file."""
        pass
    
    @abstractmethod
    def ingest_document(self, file_path: Path) -> ProcessedDocument:
        """Ingest a single document and return processed document."""
        pass


class PDFDocumentIngestor(DocumentIngestor):
    """Handles PDF document ingestion using PyMuPDF."""
    
    def __init__(self) -> None:
        """Initialize PDF ingestor."""
        super().__init__()
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are available."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF processing")
    
    def _get_supported_types(self) -> List[DocumentType]:
        """Return supported document types."""
        return [DocumentType.PDF]
    
    def can_ingest(self, file_path: Path) -> bool:
        """Check if file is a supported PDF."""
        return file_path.suffix.lower() == ".pdf" and file_path.exists()
    
    def ingest_document(self, file_path: Path) -> ProcessedDocument:
        """Ingest PDF document and extract basic information."""
        if not self.can_ingest(file_path):
            raise ValueError(f"Cannot ingest file: {file_path}")
        
        try:
            import fitz
            
            document_id = self._generate_document_id(file_path)
            doc = fitz.open(file_path)
            
            metadata = self._extract_pdf_metadata(doc)
            doc.close()
            
            return ProcessedDocument(
                document_id=document_id,
                file_path=file_path,
                document_type=DocumentType.PDF,
                metadata=metadata,
                processing_status="ingested"
            )
            
        except Exception as e:
            logger.error(f"Error ingesting PDF {file_path}: {e}")
            raise
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path."""
        return f"pdf_{file_path.stem}_{file_path.stat().st_mtime}"
    
    def _extract_pdf_metadata(self, doc) -> dict:
        """Extract metadata from PDF document."""
        metadata = {}
        
        try:
            metadata["title"] = doc.metadata.get("title", "")
            metadata["author"] = doc.metadata.get("author", "")
            metadata["subject"] = doc.metadata.get("subject", "")
            metadata["creator"] = doc.metadata.get("creator", "")
            metadata["producer"] = doc.metadata.get("producer", "")
            metadata["page_count"] = len(doc)
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
        
        return metadata


class DocumentIngestionManager:
    """
    Manages document ingestion using appropriate ingestor for each file type.
    
    Follows Dependency Inversion Principle - depends on abstractions (DocumentIngestor).
    """
    
    def __init__(self) -> None:
        """Initialize with available ingestors."""
        self.ingestors = self._initialize_ingestors()
    
    def _initialize_ingestors(self) -> List[DocumentIngestor]:
        """Initialize available document ingestors."""
        ingestors = []
        
        try:
            ingestors.append(PDFDocumentIngestor())
        except ImportError:
            logger.warning("PDF ingestion not available - PyMuPDF not installed")
        
        return ingestors
    
    def ingest_documents(self, file_paths: List[Path]) -> List[ProcessedDocument]:
        """Ingest multiple documents using appropriate ingestors."""
        processed_documents = []
        
        for file_path in file_paths:
            try:
                document = self._ingest_single_document(file_path)
                processed_documents.append(document)
                logger.info(f"Successfully ingested: {file_path}")
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                continue
        
        return processed_documents
    
    def _ingest_single_document(self, file_path: Path) -> ProcessedDocument:
        """Ingest a single document using appropriate ingestor."""
        for ingestor in self.ingestors:
            if ingestor.can_ingest(file_path):
                return ingestor.ingest_document(file_path)
        
        raise ValueError(f"No suitable ingestor found for file: {file_path}") 