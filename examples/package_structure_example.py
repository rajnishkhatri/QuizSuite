#!/usr/bin/env python3
"""
Package Structure Example

This example demonstrates the structure and usage of the document processing package.
"""

import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.document_processing.model import (
    DocumentType,
    ModalityType,
    DocumentChunk,
    ProcessedDocument,
    QuizConfig
)

from backend.document_processing.state import (
    BaseState,
    IngestNodeState,
    ProcessNodeState,
    EndNodeState
)

from backend.document_processing.node import (
    DocumentProcessingNodes,
    DocumentProcessingGraph,
    create_document_processing_graph
)

def demonstrate_package_structure():
    """Demonstrate the package structure and basic usage."""
    
    print("📦 Document Processing Package Structure")
    print("=" * 50)
    
    # Demonstrate model classes
    print("\n📋 Models:")
    print(f"  - DocumentType: {DocumentType}")
    print(f"  - ModalityType: {ModalityType}")
    print(f"  - DocumentChunk: {DocumentChunk}")
    print(f"  - ProcessedDocument: {ProcessedDocument}")
    print(f"  - QuizConfig: {QuizConfig}")
    
    # Demonstrate state classes
    print("\n🔧 States:")
    print(f"  - BaseState: {BaseState}")
    print(f"  - IngestNodeState: {IngestNodeState}")
    print(f"  - ProcessNodeState: {ProcessNodeState}")
    print(f"  - EndNodeState: {EndNodeState}")
    
    # Demonstrate node classes
    print("\n⚙️  Nodes:")
    print(f"  - DocumentProcessingNodes: {DocumentProcessingNodes}")
    print(f"  - DocumentProcessingGraph: {DocumentProcessingGraph}")
    
    # Create example instances
    print("\n🎯 Example Instances:")
    
    # Create base state
    base_state = BaseState()
    print(f"  - BaseState: {base_state}")
    
    # Create ingest state
    ingest_state = IngestNodeState()
    print(f"  - IngestNodeState: {ingest_state}")
    
    # Create process state
    process_state = ProcessNodeState()
    print(f"  - ProcessNodeState: {process_state}")
    
    # Create end state
    end_state = EndNodeState()
    print(f"  - EndNodeState: {end_state}")
    
    # Create document chunk
    chunk = DocumentChunk(
        content="Example chunk content",
        chunk_type=ModalityType.TEXT,
        chunk_index=0,
        metadata={"source": "example"}
    )
    print(f"  - DocumentChunk: {chunk}")
    
    # Create processed document
    processed_doc = ProcessedDocument(
        path="example.pdf",
        content="Example document content",
        document_type=DocumentType.PDF,
        processing_status="processed",
        chunks=[chunk]
    )
    print(f"  - ProcessedDocument: {processed_doc}")
    
    print("\n✅ Package structure demonstration completed!")

if __name__ == "__main__":
    demonstrate_package_structure() 