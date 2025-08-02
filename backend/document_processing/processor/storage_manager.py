"""
Storage Manager Module

This module handles the storage and retrieval of document chunks using LangChain and ChromaDB.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import time

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ChromaDBEmbeddingFunction:
    """
    Custom embedding function wrapper for ChromaDB compatibility.
    """
    
    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        """
        Initialize the embedding function wrapper.
        
        Args:
            embedding_model: HuggingFace embeddings model
        """
        self.embedding_model = embedding_model
    
    def __call__(self, input: Union[List[str], str]) -> List[List[float]]:
        """
        Generate embeddings for input text.
        
        Args:
            input: Text or list of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if isinstance(input, str):
            input = [input]
        
        # Generate embeddings using the HuggingFace model
        embeddings = self.embedding_model.embed_documents(input)
        return embeddings


class StorageManager:
    """
    Manages storage and indexing of embedded chunks using LangChain and ChromaDB.
    
    Features:
    - Vector database storage (ChromaDB via LangChain)
    - Metadata indexing
    - Search capabilities
    - Collection management
    """
    
    def __init__(self, 
                 persist_directory: Optional[Path] = None,
                 collection_name: str = "document_chunks",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the storage manager.
        
        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection to store chunks
            embedding_model: Name of the embedding model to use
        """
        self.persist_directory = persist_directory or Path("storage/vector_db")
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize LangChain embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 32}
        )
        
        # Create custom embedding function for ChromaDB
        self.embedding_function = ChromaDBEmbeddingFunction(self.embeddings)
        
        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB with persist directory: {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        # Initialize LangChain vectorstore
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        
        logger.info(f"Storage manager initialized with collection: {self.collection_name}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Get existing collection or create a new one.
        
        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with multimodal embeddings"},
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def store_chunks(self, embedded_chunks: List[Dict[str, Any]], 
                    document_id: str) -> Dict[str, Any]:
        """
        Store embedded chunks in the vector database.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary with storage information
        """
        logger.info(f"Storing {len(embedded_chunks)} chunks for document: {document_id}")
        
        if not embedded_chunks:
            logger.warning("No chunks to store")
            return {"stored_chunks": 0, "errors": []}
        
        # Prepare data for storage
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(embedded_chunks):
            # Get document content
            content = chunk.get('content', '')
            if not content:
                logger.warning(f"Chunk {i} has no content, skipping")
                continue
            
            # Create LangChain Document
            doc = Document(
                page_content=content,
                metadata={
                    'document_id': document_id,
                    'chunk_index': chunk.get('chunk_index', i),
                    'chunk_type': chunk.get('type', 'unknown'),
                    'modality': chunk.get('metadata', {}).get('modality', 'unknown'),
                    'embedding_type': chunk.get('embedding_metadata', {}).get('embedding_type', 'unknown'),
                    'length': len(content),
                    'page_number': chunk.get('metadata', {}).get('page_number', 0),
                    'embedding_model': chunk.get('embedding_metadata', {}).get('model', 'unknown'),
                    'embedding_dimension': chunk.get('embedding_metadata', {}).get('dimension', 0)
                }
            )
            
            # Add chunk-specific metadata
            chunk_metadata = chunk.get('metadata', {})
            for key, value in chunk_metadata.items():
                if key not in doc.metadata:
                    doc.metadata[key] = value
            
            documents.append(doc)
        
        # Store in vectorstore
        try:
            # Add documents to vectorstore
            ids = self.vectorstore.add_documents(documents)
            
            logger.info(f"Successfully stored {len(ids)} chunks in collection")
            
            return {
                "stored_chunks": len(ids),
                "document_id": document_id,
                "collection_name": self.collection_name,
                "embedding_types": list(set([doc.metadata.get('embedding_type', 'unknown') for doc in documents])),
                "chunk_types": list(set([doc.metadata.get('chunk_type', 'unknown') for doc in documents]))
            }
            
        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            return {
                "stored_chunks": 0,
                "errors": [str(e)],
                "document_id": document_id
            }
    
    def search_chunks(self, 
                     query: str,
                     n_results: int = 10,
                     filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using text query.
        
        Args:
            query: Text query to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results
        """
        logger.info(f"Searching for: '{query}' with {n_results} results")
        
        try:
            # Perform search using LangChain
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=n_results,
                filter=filter_metadata
            )
            
            # Format results
            search_results = []
            for doc, score in docs_and_scores:
                result = {
                    'document': doc.page_content,
                    'metadata': doc.metadata,
                    'distance': score
                }
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            return []
    
    def search_by_embedding(self, 
                           embedding: List[float],
                           n_results: int = 10,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding vector.
        
        Args:
            embedding: Embedding vector to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results
        """
        logger.info(f"Searching by embedding with {n_results} results")
        
        try:
            # Convert embedding to numpy array
            embedding_array = np.array(embedding).reshape(1, -1)
            
            # Perform search using ChromaDB directly
            results = self.collection.query(
                query_embeddings=embedding_array.tolist(),
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} search results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching by embedding: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand structure
            sample_results = self.collection.get(limit=1)
            metadata_keys = set()
            if sample_results['metadatas']:
                metadata_keys = set(sample_results['metadatas'][0].keys())
            
            stats = {
                'collection_name': self.collection_name,
                'total_chunks': count,
                'metadata_keys': list(metadata_keys),
                'persist_directory': str(self.persist_directory)
            }
            
            # Get chunk type distribution if available
            if count > 0:
                all_results = self.collection.get(limit=count)
                chunk_types = {}
                embedding_types = {}
                
                for metadata in all_results['metadatas']:
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    embedding_type = metadata.get('embedding_type', 'unknown')
                    
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    embedding_types[embedding_type] = embedding_types.get(embedding_type, 0) + 1
                
                stats['chunk_type_distribution'] = chunk_types
                stats['embedding_type_distribution'] = embedding_types
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def delete_document_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Dictionary with deletion information
        """
        logger.info(f"Deleting chunks for document: {document_id}")
        
        try:
            # Get chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                logger.info(f"No chunks found for document: {document_id}")
                return {"deleted_chunks": 0, "document_id": document_id}
            
            # Delete chunks
            self.collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            logger.info(f"Deleted {deleted_count} chunks for document: {document_id}")
            
            return {
                "deleted_chunks": deleted_count,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            return {
                "deleted_chunks": 0,
                "error": str(e),
                "document_id": document_id
            }
    
    def export_collection(self, output_dir: Path) -> Dict[str, Any]:
        """
        Export collection data to files.
        
        Args:
            output_dir: Directory to export data
            
        Returns:
            Dictionary with export information
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get all data from collection
            count = self.collection.count()
            if count == 0:
                logger.warning("Collection is empty, nothing to export")
                return {"exported_chunks": 0}
            
            results = self.collection.get(limit=count)
            
            # Export to JSON
            export_data = {
                'collection_name': self.collection_name,
                'total_chunks': count,
                'chunks': []
            }
            
            for i in range(len(results['ids'])):
                chunk_data = {
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'embedding': results['embeddings'][i].tolist() if 'embeddings' in results else None
                }
                export_data['chunks'].append(chunk_data)
            
            # Save to file
            export_file = output_dir / f"{self.collection_name}_export.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {count} chunks to {export_file}")
            
            return {
                "exported_chunks": count,
                "export_file": str(export_file),
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return {
                "exported_chunks": 0,
                "error": str(e)
            }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        collection_stats = self.get_collection_stats()
        
        # Get file system stats
        try:
            total_size = sum(f.stat().st_size for f in self.persist_directory.rglob('*') if f.is_file())
            file_count = len(list(self.persist_directory.rglob('*')))
        except Exception as e:
            total_size = 0
            file_count = 0
            logger.warning(f"Could not calculate file system stats: {e}")
        
        stats = {
            'collection_stats': collection_stats,
            'file_system': {
                'persist_directory': str(self.persist_directory),
                'total_size_bytes': total_size,
                'file_count': file_count
            },
            'storage_manager': {
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model
            }
        }
        
        return stats 