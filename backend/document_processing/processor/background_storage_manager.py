"""
Background Storage Manager Module

This module handles background storage of document chunks using threading,
allowing the pipeline to continue without waiting for storage operations.
"""

import logging
import threading
import queue
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .storage_manager import StorageManager

logger = logging.getLogger(__name__)


class BackgroundStorageManager:
    """
    Manages background storage of document chunks using threading.
    
    Features:
    - Asynchronous chunk storage
    - Queue-based storage operations
    - Background thread processing
    - Storage status tracking
    - Error handling and retry logic
    """
    
    def __init__(self, 
                 storage_manager: StorageManager,
                 max_workers: int = 2,
                 queue_size: int = 100,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the background storage manager.
        
        Args:
            storage_manager: The underlying storage manager
            max_workers: Maximum number of background threads
            queue_size: Maximum size of the storage queue
            retry_attempts: Number of retry attempts for failed storage
            retry_delay: Delay between retry attempts in seconds
        """
        self.storage_manager = storage_manager
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Storage queue and thread management
        self.storage_queue = queue.Queue(maxsize=queue_size)
        self.storage_threads = []
        self.storage_status = {}  # Track storage status by document_id
        self.storage_errors = {}  # Track storage errors by document_id
        self.is_running = False
        
        # Thread pool for parallel storage operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Start background storage threads
        self._start_storage_threads()
        
        logger.info(f"✅ Background storage manager initialized with {max_workers} workers")
    
    def _start_storage_threads(self):
        """Start background storage threads."""
        self.is_running = True
        
        for i in range(self.max_workers):
            thread = threading.Thread(
                target=self._storage_worker,
                name=f"StorageWorker-{i}",
                daemon=True
            )
            thread.start()
            self.storage_threads.append(thread)
        
        logger.info(f"🚀 Started {self.max_workers} background storage threads")
    
    def _storage_worker(self):
        """Background worker thread for processing storage operations."""
        while self.is_running:
            try:
                # Get storage task from queue with timeout
                task = self.storage_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                # Process storage task
                self._process_storage_task(task)
                
                # Mark task as done
                self.storage_queue.task_done()
                
            except queue.Empty:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"❌ Storage worker error: {e}")
                continue
    
    def _process_storage_task(self, task: Dict[str, Any]):
        """Process a storage task."""
        document_id = task.get('document_id')
        chunks = task.get('chunks', [])
        callback = task.get('callback')
        
        logger.info(f"🔄 Processing storage task for document: {document_id}")
        
        # Update status to processing
        self.storage_status[document_id] = {
            'status': 'processing',
            'start_time': time.time(),
            'chunk_count': len(chunks)
        }
        
        # Attempt storage with retry logic
        for attempt in range(self.retry_attempts):
            try:
                # Store chunks using the underlying storage manager
                storage_result = self.storage_manager.store_chunks(chunks, document_id)
                
                # Update status to completed
                self.storage_status[document_id] = {
                    'status': 'completed',
                    'start_time': self.storage_status[document_id]['start_time'],
                    'end_time': time.time(),
                    'chunk_count': len(chunks),
                    'stored_chunks': storage_result.get('stored_chunks', 0),
                    'result': storage_result
                }
                
                # Clear any previous errors
                if document_id in self.storage_errors:
                    del self.storage_errors[document_id]
                
                logger.info(f"✅ Storage completed for document: {document_id}")
                
                # Call callback if provided
                if callback:
                    try:
                        callback(document_id, storage_result)
                    except Exception as e:
                        logger.error(f"❌ Callback error for {document_id}: {e}")
                
                return
                
            except Exception as e:
                logger.warning(f"⚠️ Storage attempt {attempt + 1} failed for {document_id}: {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    # Final attempt failed
                    self.storage_status[document_id] = {
                        'status': 'failed',
                        'start_time': self.storage_status[document_id]['start_time'],
                        'end_time': time.time(),
                        'chunk_count': len(chunks),
                        'error': str(e)
                    }
                    
                    # Store error
                    self.storage_errors[document_id] = {
                        'error': str(e),
                        'attempts': self.retry_attempts,
                        'timestamp': time.time()
                    }
                    
                    logger.error(f"❌ Storage failed for document: {document_id} after {self.retry_attempts} attempts")
    
    def store_chunks_async(self, 
                          chunks: List[Dict[str, Any]], 
                          document_id: str,
                          callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Queue chunks for background storage.
        
        Args:
            chunks: List of chunks to store
            document_id: Unique identifier for the document
            callback: Optional callback function to call when storage completes
            
        Returns:
            Dictionary with queuing information
        """
        if not chunks:
            logger.warning(f"No chunks to store for document: {document_id}")
            return {
                "queued": False,
                "reason": "no_chunks",
                "document_id": document_id
            }
        
        # Create storage task
        task = {
            'document_id': document_id,
            'chunks': chunks,
            'callback': callback,
            'timestamp': time.time()
        }
        
        try:
            # Add task to queue
            self.storage_queue.put(task, timeout=5.0)
            
            # Update status to queued
            self.storage_status[document_id] = {
                'status': 'queued',
                'queue_time': time.time(),
                'chunk_count': len(chunks)
            }
            
            logger.info(f"📥 Queued {len(chunks)} chunks for background storage: {document_id}")
            
            return {
                "queued": True,
                "document_id": document_id,
                "chunk_count": len(chunks),
                "queue_size": self.storage_queue.qsize()
            }
            
        except queue.Full:
            logger.error(f"❌ Storage queue is full, cannot queue chunks for: {document_id}")
            return {
                "queued": False,
                "reason": "queue_full",
                "document_id": document_id
            }
        except Exception as e:
            logger.error(f"❌ Error queuing storage task for {document_id}: {e}")
            return {
                "queued": False,
                "reason": "error",
                "error": str(e),
                "document_id": document_id
            }
    
    def get_storage_status(self, document_id: str) -> Dict[str, Any]:
        """Get storage status for a specific document."""
        return self.storage_status.get(document_id, {
            'status': 'unknown',
            'document_id': document_id
        })
    
    def get_all_storage_status(self) -> Dict[str, Any]:
        """Get status of all storage operations."""
        return {
            'storage_status': self.storage_status,
            'storage_errors': self.storage_errors,
            'queue_size': self.storage_queue.qsize(),
            'active_threads': len([t for t in self.storage_threads if t.is_alive()]),
            'total_threads': len(self.storage_threads)
        }
    
    def wait_for_completion(self, document_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """
        Wait for storage completion of a specific document.
        
        Args:
            document_id: Document ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Storage status
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_storage_status(document_id)
            
            if status['status'] in ['completed', 'failed']:
                return status
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
        
        # Timeout reached
        return {
            'status': 'timeout',
            'document_id': document_id,
            'timeout': timeout
        }
    
    def wait_for_all_completion(self, timeout: float = 600.0) -> Dict[str, Any]:
        """
        Wait for all queued storage operations to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            Summary of all storage operations
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if queue is empty and all operations are complete
            if self.storage_queue.empty():
                all_completed = True
                for doc_id, status in self.storage_status.items():
                    if status['status'] not in ['completed', 'failed']:
                        all_completed = False
                        break
                
                if all_completed:
                    return self.get_all_storage_status()
            
            time.sleep(0.5)
        
        # Timeout reached
        return {
            'status': 'timeout',
            'timeout': timeout,
            'current_status': self.get_all_storage_status()
        }
    
    def shutdown(self, wait_for_completion: bool = True, timeout: float = 300.0):
        """
        Shutdown the background storage manager.
        
        Args:
            wait_for_completion: Whether to wait for queued operations to complete
            timeout: Maximum time to wait for completion
        """
        logger.info("🛑 Shutting down background storage manager...")
        
        # Stop accepting new tasks
        self.is_running = False
        
        if wait_for_completion:
            logger.info("⏳ Waiting for queued operations to complete...")
            self.wait_for_all_completion(timeout)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Send shutdown signals to storage threads
        for _ in self.storage_threads:
            self.storage_queue.put(None)
        
        # Wait for threads to finish
        for thread in self.storage_threads:
            thread.join(timeout=5.0)
        
        logger.info("✅ Background storage manager shutdown complete")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = self.get_all_storage_status()
        
        # Calculate additional statistics
        completed_count = sum(1 for status in self.storage_status.values() 
                            if status.get('status') == 'completed')
        failed_count = sum(1 for status in self.storage_status.values() 
                          if status.get('status') == 'failed')
        processing_count = sum(1 for status in self.storage_status.values() 
                             if status.get('status') == 'processing')
        queued_count = sum(1 for status in self.storage_status.values() 
                          if status.get('status') == 'queued')
        
        stats.update({
            'completed_operations': completed_count,
            'failed_operations': failed_count,
            'processing_operations': processing_count,
            'queued_operations': queued_count,
            'total_operations': len(self.storage_status)
        })
        
        return stats 