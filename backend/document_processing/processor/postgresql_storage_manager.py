"""
PostgreSQL Storage Manager for Chunks

This module handles the storage and retrieval of quality filtered chunks using PostgreSQL.
Provides efficient storage, indexing, and retrieval capabilities for the cache manager.
"""

import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Text, Integer, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class ChunkModel(Base):
    """SQLAlchemy model for chunks table."""
    __tablename__ = 'chunks'
    
    chunk_id = Column(String(255), primary_key=True)
    document_id = Column(String(255), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    chunk_metadata = Column(JSON, nullable=True)  # Changed from 'metadata' to 'chunk_metadata'
    quality_score = Column(Float, default=0.0)
    source_document = Column(String(500), nullable=True)
    chunk_type = Column(String(100), nullable=True)
    modality = Column(String(100), nullable=True)
    content_length = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProcessingStatusModel(Base):
    """SQLAlchemy model for processing status table."""
    __tablename__ = 'processing_status'
    
    id = Column(String(255), primary_key=True)
    current_step = Column(String(100), nullable=False)
    status_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PostgreSQLStorageManager:
    """
    Manages storage and retrieval of quality filtered chunks using PostgreSQL.
    
    Features:
    - Efficient chunk storage with indexing
    - Metadata storage in JSON format
    - Processing status tracking
    - Connection pooling for performance
    - Automatic table creation
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "quizsuite",
                 username: str = "postgres",
                 password: str = "postgres",
                 min_connections: int = 1,
                 max_connections: int = 10):
        """
        Initialize the PostgreSQL storage manager.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            username: Database username
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        
        # Connection pool
        self.pool = None
        self.engine = None
        self.SessionLocal = None
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"✅ PostgreSQL storage manager initialized for database: {database}")
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Create SQLAlchemy engine
            connection_string = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(connection_string, pool_pre_ping=True)
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create connection pool
            self.pool = SimpleConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            
            logger.info("✅ Database tables created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _get_connection(self):
        """Get a connection from the pool."""
        return self.pool.getconn()
    
    def _return_connection(self, conn):
        """Return a connection to the pool."""
        self.pool.putconn(conn)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def store_chunks(self, chunks: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """
        Store quality filtered chunks in PostgreSQL.
        
        Args:
            chunks: List of chunks to store
            document_id: Document identifier
            
        Returns:
            Dictionary with storage information
        """
        if not chunks:
            logger.warning("No chunks to store")
            return {"stored": False, "reason": "no_chunks"}
        
        try:
            session = self.SessionLocal()
            stored_count = 0
            errors = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate chunk ID and content hash
                    chunk_id = f"{document_id}_chunk_{i}_{hash(chunk.get('content', '')) % 100000000:08x}"
                    content_hash = self._generate_content_hash(chunk.get('content', ''))
                    
                    # Check if chunk already exists
                    existing_chunk = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
                    if existing_chunk:
                        logger.debug(f"Chunk {chunk_id} already exists, skipping")
                        continue
                    
                    # Create new chunk record
                    chunk_record = ChunkModel(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=i,
                        content=chunk.get('content', ''),
                        content_hash=content_hash,
                        chunk_metadata=chunk.get('metadata', {}),  # Updated column name
                        quality_score=chunk.get('quality_score', 0.0),
                        source_document=chunk.get('source_document', 'unknown'),
                        chunk_type=chunk.get('type', 'unknown'),
                        modality=chunk.get('metadata', {}).get('modality', 'unknown'),
                        content_length=len(chunk.get('content', ''))
                    )
                    
                    session.add(chunk_record)
                    stored_count += 1
                    
                except Exception as e:
                    error_msg = f"Error storing chunk {i}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Commit all changes
            session.commit()
            session.close()
            
            logger.info(f"✅ Stored {stored_count} chunks in PostgreSQL for document: {document_id}")
            
            return {
                "stored": True,
                "stored_count": stored_count,
                "total_chunks": len(chunks),
                "document_id": document_id,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to store chunks in PostgreSQL: {e}")
            if session:
                session.rollback()
                session.close()
            return {"stored": False, "error": str(e)}
    
    def load_chunks(self, document_id: Optional[str] = None, 
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load chunks from PostgreSQL.
        
        Args:
            document_id: Optional document ID filter
            limit: Maximum number of chunks to return
            offset: Number of chunks to skip
            
        Returns:
            List of chunks
        """
        try:
            session = self.SessionLocal()
            
            # Build query
            query = session.query(ChunkModel)
            
            if document_id:
                query = query.filter_by(document_id=document_id)
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            # Order by chunk_index
            query = query.order_by(ChunkModel.chunk_index)
            
            # Execute query
            chunk_records = query.all()
            
            # Convert to dictionary format
            chunks = []
            for record in chunk_records:
                chunk = {
                    'chunk_id': record.chunk_id,
                    'document_id': record.document_id,
                    'chunk_index': record.chunk_index,
                    'content': record.content,
                    'metadata': record.chunk_metadata or {},  # Updated column name
                    'quality_score': record.quality_score,
                    'source_document': record.source_document,
                    'type': record.chunk_type,
                    'content_length': record.content_length,
                    'created_at': record.created_at.isoformat() if record.created_at else None
                }
                chunks.append(chunk)
            
            session.close()
            
            logger.info(f"✅ Loaded {len(chunks)} chunks from PostgreSQL")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to load chunks from PostgreSQL: {e}")
            if session:
                session.close()
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk data or None if not found
        """
        try:
            session = self.SessionLocal()
            record = session.query(ChunkModel).filter_by(chunk_id=chunk_id).first()
            
            if not record:
                session.close()
                return None
            
            chunk = {
                'chunk_id': record.chunk_id,
                'document_id': record.document_id,
                'chunk_index': record.chunk_index,
                'content': record.content,
                'metadata': record.chunk_metadata or {},  # Updated column name
                'quality_score': record.quality_score,
                'source_document': record.source_document,
                'type': record.chunk_type,
                'content_length': record.content_length,
                'created_at': record.created_at.isoformat() if record.created_at else None
            }
            
            session.close()
            return chunk
            
        except Exception as e:
            logger.error(f"❌ Failed to get chunk {chunk_id}: {e}")
            if session:
                session.close()
            return None
    
    def delete_chunks(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with deletion information
        """
        try:
            session = self.SessionLocal()
            
            # Count chunks before deletion
            chunk_count = session.query(ChunkModel).filter_by(document_id=document_id).count()
            
            # Delete chunks
            deleted_count = session.query(ChunkModel).filter_by(document_id=document_id).delete()
            
            session.commit()
            session.close()
            
            logger.info(f"✅ Deleted {deleted_count} chunks for document: {document_id}")
            
            return {
                "deleted": True,
                "deleted_count": deleted_count,
                "total_chunks": chunk_count,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to delete chunks for document {document_id}: {e}")
            if session:
                session.rollback()
                session.close()
            return {"deleted": False, "error": str(e)}
    
    def get_chunk_statistics(self, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored chunks.
        
        Args:
            document_id: Optional document ID filter
            
        Returns:
            Dictionary with statistics
        """
        try:
            session = self.SessionLocal()
            
            # Build base query
            query = session.query(ChunkModel)
            
            if document_id:
                query = query.filter_by(document_id=document_id)
            
            # Get total count
            total_chunks = query.count()
            
            # Get statistics
            from sqlalchemy import func
            stats = session.query(
                ChunkModel.chunk_type,
                ChunkModel.modality,
                func.count(ChunkModel.chunk_id).label('count'),
                func.avg(ChunkModel.quality_score).label('avg_quality'),
                func.avg(ChunkModel.content_length).label('avg_length')
            ).group_by(ChunkModel.chunk_type, ChunkModel.modality).all()
            
            # Get document count
            document_count = session.query(ChunkModel.document_id).distinct().count()
            
            session.close()
            
            # Format statistics
            chunk_types = {}
            modalities = {}
            
            for stat in stats:
                chunk_type = stat.chunk_type or 'unknown'
                modality = stat.modality or 'unknown'
                
                if chunk_type not in chunk_types:
                    chunk_types[chunk_type] = 0
                chunk_types[chunk_type] += stat.count
                
                if modality not in modalities:
                    modalities[modality] = 0
                modalities[modality] += stat.count
            
            return {
                "total_chunks": total_chunks,
                "document_count": document_count,
                "chunk_types": chunk_types,
                "modalities": modalities,
                "document_id": document_id
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get chunk statistics: {e}")
            if session:
                session.close()
            return {"error": str(e)}
    
    def update_processing_status(self, status_data: Dict[str, Any]) -> bool:
        """
        Update processing status in PostgreSQL.
        
        Args:
            status_data: Status information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.SessionLocal()
            
            # Create or update status record
            status_record = ProcessingStatusModel(
                id="current_processing",
                current_step=status_data.get('current_step', 'unknown'),
                status_data=status_data
            )
            
            # Use merge to handle upsert
            session.merge(status_record)
            session.commit()
            session.close()
            
            logger.info(f"✅ Updated processing status: {status_data.get('current_step', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update processing status: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_processing_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current processing status.
        
        Returns:
            Status data or None if not found
        """
        try:
            session = self.SessionLocal()
            record = session.query(ProcessingStatusModel).filter_by(id="current_processing").first()
            
            if not record:
                session.close()
                return None
            
            status_data = record.status_data or {}
            status_data['current_step'] = record.current_step
            status_data['updated_at'] = record.updated_at.isoformat() if record.updated_at else None
            
            session.close()
            return status_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get processing status: {e}")
            if session:
                session.close()
            return None
    
    def close(self):
        """Close database connections."""
        if self.pool:
            self.pool.closeall()
        logger.info("✅ PostgreSQL storage manager connections closed") 