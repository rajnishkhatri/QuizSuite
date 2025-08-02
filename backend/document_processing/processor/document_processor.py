"""
Document Processor Module

This module handles the processing of documents, including content extraction,
chunking, embedding generation, and storage using LangChain.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .chunking_strategy import ModalityAwareChunkingStrategy, SimpleChunkingStrategy
from .content_extractor import ContentExtractor
from .embedding_manager import EmbeddingManager
from .storage_manager import StorageManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document processor that handles content extraction, chunking, embedding generation, and storage.
    
    This processor integrates:
    - Content extraction (images, figures, tables, code)
    - Modality-aware chunking
    - Embedding generation (LangChain)
    - Vector database storage (ChromaDB via LangChain)
    - Metadata enrichment
    - Text cleaning and preprocessing
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap_size: int = 200,
                 extract_content: bool = True,
                 generate_embeddings: bool = True,
                 store_chunks: bool = True,
                 output_dir: Optional[Path] = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 collection_name: str = "document_chunks",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each chunk in characters
            overlap_size: Overlap between chunks in characters
            extract_content: Whether to extract images, figures, tables, and code
            generate_embeddings: Whether to generate embeddings for chunks
            store_chunks: Whether to store chunks in vector database
            output_dir: Directory to save extracted content
            embedding_model: Name of the embedding model to use
            collection_name: Name of the collection for vector storage
            config: Configuration dictionary for LangChain settings
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.extract_content = extract_content
        self.generate_embeddings = generate_embeddings
        self.store_chunks = store_chunks
        self.output_dir = output_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.config = config or {}
        
        # Initialize chunking strategy
        if extract_content:
            self.chunking_strategy = ModalityAwareChunkingStrategy(
                chunk_size=chunk_size,
                overlap_size=overlap_size,
                extract_content=True,
                output_dir=output_dir
            )
        else:
            self.chunking_strategy = SimpleChunkingStrategy(
                chunk_size=chunk_size,
                overlap_size=overlap_size
            )
        
        # Initialize content extractor for standalone extraction
        self.content_extractor = None
        if extract_content:
            self.content_extractor = ContentExtractor(output_dir)
        
        # Initialize embedding manager (LangChain-based)
        self.embedding_manager = None
        if generate_embeddings:
            # Get LangChain settings from config
            langchain_settings = self.config.get('langchain_settings', {})
            embedding_device = langchain_settings.get('embedding_device', 'cpu')
            device_type = langchain_settings.get('device_type', 'mps')
            embedding_batch_size = langchain_settings.get('embedding_batch_size', 32)
            
            self.embedding_manager = EmbeddingManager(
                model_name=embedding_model,
                device=embedding_device,
                device_type=device_type,
                batch_size=embedding_batch_size
            )
        
        # Initialize storage manager (LangChain-based)
        self.storage_manager = None
        if store_chunks:
            self.storage_manager = StorageManager(
                collection_name=collection_name,
                embedding_model=embedding_model
            )
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document with content extraction, chunking, embedding generation, and storage.
        
        Args:
            document: Document to process
            
        Returns:
            Processed document with chunks, embeddings, and storage information
        """
        logger.info(f"Processing document: {document.get('path', 'unknown')}")
        
        try:
            # Step 1: Extract content (images, figures, tables, code)
            extracted_content = self._extract_content(document)
            
            # Step 2: Clean and preprocess text
            cleaned_document = self._clean_document(document)
            
            # Step 3: Create chunks using modality-aware strategy (pass extracted content)
            chunks = self.chunking_strategy.chunk_document(cleaned_document, extracted_content)
            
            # Step 4: Enrich chunks with metadata
            enriched_chunks = self._enrich_chunks(chunks, extracted_content)
            
            # Step 5: Generate embeddings (LangChain-based)
            embedded_chunks = self._generate_embeddings(enriched_chunks)
            
            # Step 6: Store chunks in vector database (LangChain-based)
            storage_info = self._store_chunks(embedded_chunks, document)
            
            # Step 7: Create processing summary
            processing_summary = self._create_processing_summary(
                document, embedded_chunks, extracted_content, storage_info
            )
            
            # Step 8: Prepare final result
            result = {
                'original_document': document,
                'processed_document': cleaned_document,
                'chunks': embedded_chunks,
                'extracted_content': extracted_content,
                'storage_info': storage_info,
                'processing_summary': processing_summary,
                'metadata': {
                    'total_chunks': len(embedded_chunks),
                    'chunks_by_type': self._count_chunks_by_type(embedded_chunks),
                    'chunks_by_modality': self._count_chunks_by_modality(embedded_chunks),
                    'content_extraction_enabled': self.extract_content,
                    'embedding_generation_enabled': self.generate_embeddings,
                    'storage_enabled': self.store_chunks,
                    'chunk_size': self.chunk_size,
                    'overlap_size': self.overlap_size
                }
            }
            
            logger.info(f"Document processing completed successfully")
            logger.info(f"  - Total chunks: {len(embedded_chunks)}")
            logger.info(f"  - Content extraction: {'enabled' if self.extract_content else 'disabled'}")
            logger.info(f"  - Embedding generation: {'enabled' if self.generate_embeddings else 'disabled'}")
            logger.info(f"  - Storage: {'enabled' if self.store_chunks else 'disabled'}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def _extract_content(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract content from the document.
        
        Args:
            document: Document to extract content from
            
        Returns:
            Extracted content dictionary or None if extraction is disabled
        """
        if not self.extract_content or not self.content_extractor:
            logger.info("Content extraction is disabled")
            return None
        
        try:
            pdf_path = document.get('path', '')
            if not pdf_path or not Path(pdf_path).exists():
                logger.warning(f"PDF path not found or invalid: {pdf_path}")
                return None
            
            logger.info(f"Extracting content from: {pdf_path}")
            extracted_content = self.content_extractor.extract_all_content(pdf_path)
            
            # Save extracted content if output directory is specified
            if self.output_dir:
                extracted_content = self.content_extractor.save_extracted_content(
                    extracted_content, self.output_dir
                )
            
            # Generate extraction summary
            extraction_summary = self.content_extractor.get_extraction_summary(extracted_content)
            logger.info(f"Content extraction completed: {extraction_summary}")
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error during content extraction: {e}")
            return None
    
    def _clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and preprocess the document.
        
        Args:
            document: Document to clean
            
        Returns:
            Cleaned document
        """
        cleaned_document = document.copy()
        content = document.get('content', '')
        
        if content:
            # Basic text cleaning
            cleaned_content = self._clean_text(content)
            cleaned_document['content'] = cleaned_content
            cleaned_document['original_length'] = len(content)
            cleaned_document['cleaned_length'] = len(cleaned_content)
        
        return cleaned_document
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess text content.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might interfere with processing
        # Keep basic punctuation and alphanumeric characters
        import re
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _enrich_chunks(self, chunks: List[Dict[str, Any]], 
                      extracted_content: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich chunks with additional metadata and content information.
        
        Args:
            chunks: List of chunks to enrich
            extracted_content: Extracted content dictionary
            
        Returns:
            Enriched chunks
        """
        enriched_chunks = []
        
        for chunk in chunks:
            enriched_chunk = chunk.copy()
            
            # Add processing metadata
            enriched_chunk['metadata'] = enriched_chunk.get('metadata', {})
            enriched_chunk['metadata'].update({
                'processed': True,
                'content_extraction_enabled': self.extract_content,
                'chunking_strategy': 'modality_aware' if self.extract_content else 'simple'
            })
            
            # Add content-specific metadata if available
            if extracted_content and chunk.get('type') in ['image', 'table', 'code', 'figure']:
                chunk_type = chunk.get('type')
                chunk_index = chunk.get('chunk_index', 0)
                
                # Find corresponding extracted content
                if chunk_type == 'image':
                    for img in extracted_content.get('images', []):
                        if img.get('image_index') == chunk_index:
                            enriched_chunk['metadata'].update({
                                'ocr_success': img.get('ocr_success', False),
                                'image_dimensions': f"{img.get('width', 0)}x{img.get('height', 0)}",
                                'image_size_bytes': img.get('size_bytes', 0)
                            })
                            break
                
                elif chunk_type == 'table':
                    for tbl in extracted_content.get('tables', []):
                        if tbl.get('table_index') == chunk_index:
                            enriched_chunk['metadata'].update({
                                'table_rows': tbl.get('rows', 0),
                                'table_columns': tbl.get('columns', 0),
                                'table_cells': tbl.get('total_cells', 0)
                            })
                            break
                
                elif chunk_type == 'code':
                    for code in extracted_content.get('code_chunks', []):
                        if code.get('chunk_index') == chunk_index:
                            enriched_chunk['metadata'].update({
                                'code_type': code.get('type', 'unknown'),
                                'code_lines': code.get('lines', 0),
                                'code_characters': code.get('characters', 0)
                            })
                            break
            
            enriched_chunks.append(enriched_chunk)
        
        return enriched_chunks
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks using LangChain.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            List of chunks with embeddings
        """
        if not self.generate_embeddings or not self.embedding_manager:
            logger.info("Embedding generation is disabled")
            return chunks
        
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks using LangChain")
            embedded_chunks = self.embedding_manager.generate_embeddings(chunks)
            
            # Get embedding statistics
            embedding_stats = self.embedding_manager.get_embedding_stats(embedded_chunks)
            logger.info(f"Embedding generation completed: {embedding_stats}")
            
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return chunks
    
    def _store_chunks(self, chunks: List[Dict[str, Any]], document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store chunks in vector database using LangChain and ChromaDB.
        
        Args:
            chunks: List of chunks to store
            document: Original document
            
        Returns:
            Storage information
        """
        if not self.store_chunks or not self.storage_manager:
            logger.info("Storage is disabled")
            return {"stored_chunks": 0, "storage_disabled": True}
        
        try:
            # Generate document ID
            document_id = document.get('path', 'unknown').replace('/', '_').replace('.', '_')
            
            logger.info(f"Storing {len(chunks)} chunks for document: {document_id}")
            storage_info = self.storage_manager.store_chunks(chunks, document_id)
            
            logger.info(f"Storage completed: {storage_info}")
            return storage_info
            
        except Exception as e:
            logger.error(f"Error during storage: {e}")
            return {
                "stored_chunks": 0,
                "error": str(e)
            }
    
    def _create_processing_summary(self, document: Dict[str, Any], 
                                 chunks: List[Dict[str, Any]], 
                                 extracted_content: Optional[Dict[str, Any]],
                                 storage_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the processing results.
        
        Args:
            document: Original document
            chunks: Processed chunks
            extracted_content: Extracted content
            storage_info: Storage information
            
        Returns:
            Processing summary dictionary
        """
        summary = {
            'document_info': {
                'path': document.get('path', 'unknown'),
                'original_length': document.get('original_length', 0),
                'cleaned_length': document.get('cleaned_length', 0)
            },
            'chunking_info': {
                'total_chunks': len(chunks),
                'chunks_by_type': self._count_chunks_by_type(chunks),
                'chunks_by_modality': self._count_chunks_by_modality(chunks)
            },
            'content_extraction_info': {
                'enabled': self.extract_content,
                'extracted_content': extracted_content is not None
            },
            'embedding_info': {
                'enabled': self.generate_embeddings,
                'model': self.embedding_model if self.generate_embeddings else None
            },
            'storage_info': {
                'enabled': self.store_chunks,
                'collection_name': self.collection_name if self.store_chunks else None,
                'storage_results': storage_info
            }
        }
        
        # Add content extraction summary if available
        if extracted_content and self.content_extractor:
            extraction_summary = self.content_extractor.get_extraction_summary(extracted_content)
            summary['content_extraction_info']['summary'] = extraction_summary
        
        # Add chunking summary if available
        if hasattr(self.chunking_strategy, 'get_chunking_summary'):
            chunking_summary = self.chunking_strategy.get_chunking_summary(chunks)
            summary['chunking_info']['detailed_summary'] = chunking_summary
        
        # Add embedding summary if available
        if self.generate_embeddings and self.embedding_manager:
            embedding_stats = self.embedding_manager.get_embedding_stats(chunks)
            summary['embedding_info']['stats'] = embedding_stats
        
        return summary
    
    def _count_chunks_by_type(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count chunks by type."""
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk.get('type', 'unknown')
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        return type_counts
    
    def _count_chunks_by_modality(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count chunks by modality."""
        modality_counts = {}
        for chunk in chunks:
            modality = chunk.get('metadata', {}).get('modality', 'unknown')
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        return modality_counts
    
    def get_processing_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get processing statistics from a processing result.
        
        Args:
            result: Processing result dictionary
            
        Returns:
            Processing statistics
        """
        stats = {
            'total_chunks': len(result.get('chunks', [])),
            'chunks_by_type': result.get('metadata', {}).get('chunks_by_type', {}),
            'chunks_by_modality': result.get('metadata', {}).get('chunks_by_modality', {}),
            'content_extraction_enabled': result.get('metadata', {}).get('content_extraction_enabled', False),
            'embedding_generation_enabled': result.get('metadata', {}).get('embedding_generation_enabled', False),
            'storage_enabled': result.get('metadata', {}).get('storage_enabled', False),
            'processing_summary': result.get('processing_summary', {}),
            'storage_info': result.get('storage_info', {})
        }
        
        return stats 