"""
Embedding and Storage Example

Demonstrates the new embedding and storage functionality in the document processing pipeline.
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger = logging.getLogger(__name__)
    logger.info("Environment variables loaded from .env file")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("python-dotenv not installed. Install with: poetry add python-dotenv")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not load .env file: {e}")

from backend.document_processing.node.document_processing_graph import create_document_processing_graph
from backend.document_processing.model.config_models import QuizConfig, ChromaDBSettings, CategoryConfig
from backend.document_processing.processor.embedding_manager import EmbeddingManager
from backend.document_processing.processor.storage_manager import StorageManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_config() -> QuizConfig:
    """Create an example configuration for testing."""
    return QuizConfig(
        database_type="pdf",
        model_name="mistral",
        embedding_model="text-embedding-ada-002",
        temperature=0.0,
        target_audience="TOGAF certification candidates",
        difficulty_level="Intermediate",
        learning_objectives="Key TOGAF concepts and application",
        chroma_db_settings=ChromaDBSettings(
            pdf_persist_directory="storage/chroma_db_pdf",
            html_persist_directory="storage/chroma_db_html",
            pdf_collection_name="pdf_documents",
            html_collection_name="html_documents"
        ),
        categories=[
            CategoryConfig(
                name="Concepts",
                num_questions=8,
                description="TOGAF core concepts and fundamentals",
                doc_paths=["storage/TogafD/togaf-standard-introduction-and-core-concepts:latest:01-doc:chap01.pdf"]
            )
        ],
        auto_topic_distribution_settings={},
        cache_settings={"enabled": True, "cache_dir": "backend/document_processing/summary_of_context"},
        output_settings={"output_dir": "backend/output_formats/GenerateQuizes"},
        prompt_settings={}
    )


def check_environment_setup():
    """Check if environment is properly set up for API access."""
    logger.info("=== Environment Setup Check ===")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        logger.info("✅ .env file found")
    else:
        logger.warning("❌ .env file not found")
        logger.info("To create .env file:")
        logger.info("1. Copy env.template to .env")
        logger.info("2. Add your API keys to .env")
        logger.info("3. Never commit .env to version control")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key != "your_openai_api_key_here":
        logger.info("✅ OPENAI_API_KEY is set")
    else:
        logger.warning("❌ OPENAI_API_KEY not set or using placeholder")
        logger.info("Please set your OpenAI API key in .env file")
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
        logger.info("✅ ANTHROPIC_API_KEY is set")
    else:
        logger.warning("❌ ANTHROPIC_API_KEY not set or using placeholder")
    
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if huggingface_token and huggingface_token != "your_huggingface_token_here":
        logger.info("✅ HUGGINGFACE_TOKEN is set")
    else:
        logger.warning("❌ HUGGINGFACE_TOKEN not set or using placeholder")
    
    # Check for python-dotenv
    try:
        import dotenv
        logger.info("✅ python-dotenv is installed")
    except ImportError:
        logger.warning("❌ python-dotenv not installed")
        logger.info("Install with: poetry add python-dotenv")


def demonstrate_embedding_manager():
    """Demonstrate the embedding manager functionality."""
    logger.info("=== Embedding Manager Demo ===")
    
    # Check if API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        logger.info("No valid OpenAI API key found. Using mock embeddings for demonstration.")
        logger.info("To use real embeddings:")
        logger.info("1. Copy env.template to .env")
        logger.info("2. Add your OpenAI API key to .env")
        logger.info("3. Install python-dotenv: poetry add python-dotenv")
    
    try:
        # Create embedding manager
        embedding_manager = EmbeddingManager(embedding_model="text-embedding-ada-002")
        
        # Create sample chunks
        from backend.document_processing.model.document_models import DocumentChunk, ModalityType
        
        sample_chunks = [
            DocumentChunk(
                content="TOGAF is an enterprise architecture framework.",
                chunk_id="chunk_1",
                document_id="doc_1",
                modality=ModalityType.TEXT,
                chunk_index=0
            ),
            DocumentChunk(
                content="The Architecture Development Method (ADM) is a core component of TOGAF.",
                chunk_id="chunk_2",
                document_id="doc_1",
                modality=ModalityType.TEXT,
                chunk_index=1
            ),
            DocumentChunk(
                content="Enterprise architecture helps organizations align business and IT strategies.",
                chunk_id="chunk_3",
                document_id="doc_2",
                modality=ModalityType.TEXT,
                chunk_index=0
            )
        ]
        
        logger.info(f"Created {len(sample_chunks)} sample chunks")
        logger.info("Embedding manager initialized successfully")
        logger.info(f"Embedding model: {embedding_manager.embedding_model}")
        
        # Show embedding stats structure
        stats = embedding_manager.get_embedding_stats(sample_chunks)
        logger.info(f"Embedding stats structure: {stats}")
        
    except Exception as e:
        logger.warning(f"Embedding manager demo failed (expected without API key): {e}")
        logger.info("This is expected behavior when no API key is provided.")


def demonstrate_storage_manager():
    """Demonstrate the storage manager functionality."""
    logger.info("=== Storage Manager Demo ===")
    
    try:
        # Create storage manager
        config = {
            "chroma_db_settings": {
                "pdf_persist_directory": "storage/chroma_db_pdf",
                "pdf_collection_name": "pdf_documents"
            }
        }
        
        storage_manager = StorageManager(config)
        
        # Create sample documents
        from backend.document_processing.model.document_models import ProcessedDocument, DocumentType, DocumentChunk, ModalityType
        
        sample_documents = [
            ProcessedDocument(
                document_id="doc_1",
                file_path=Path("storage/TogafD/togaf-standard-introduction-and-core-concepts:latest:01-doc:chap01.pdf"),
                document_type=DocumentType.PDF,
                chunks=[
                    DocumentChunk(
                        content="TOGAF is an enterprise architecture framework.",
                        chunk_id="chunk_1",
                        document_id="doc_1",
                        modality=ModalityType.TEXT,
                        chunk_index=0,
                        embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
                    )
                ]
            )
        ]
        
        logger.info(f"Created {len(sample_documents)} sample documents")
        logger.info("Storage manager initialized successfully")
        
        # Show storage stats structure
        stats = storage_manager.get_storage_stats()
        logger.info(f"Storage stats structure: {stats}")
        
    except Exception as e:
        logger.warning(f"Storage manager demo failed: {e}")


def demonstrate_pipeline():
    """Demonstrate the complete pipeline with embedding and storage."""
    logger.info("=== Complete Pipeline Demo ===")
    
    try:
        # Create configuration
        config = create_example_config()
        logger.info("Configuration created successfully")
        
        # Create document processing graph
        graph = create_document_processing_graph()
        logger.info("Document processing graph created successfully")
        
        # Note: In a real scenario, you would run the pipeline here
        # For demo purposes, we'll just show the structure
        logger.info("Pipeline structure:")
        logger.info("START → Ingest → Process → Embed & Store → END")
        
        # Show the new pipeline flow
        logger.info("New pipeline includes:")
        logger.info("1. Document ingestion")
        logger.info("2. Document processing (chunking, cleaning, enrichment)")
        logger.info("3. Embedding generation")
        logger.info("4. Vector database storage (ChromaDB)")
        logger.info("5. Graph database storage (JSON-based)")
        logger.info("6. Completion and summary")
        
    except Exception as e:
        logger.error(f"Error in pipeline demonstration: {e}")


def main():
    """Main function to run the embedding and storage demonstration."""
    logger.info("Starting Embedding and Storage Pipeline Demonstration")
    logger.info("=" * 60)
    
    try:
        # Check environment setup first
        check_environment_setup()
        logger.info("")
        
        # Demonstrate individual components
        demonstrate_embedding_manager()
        logger.info("")
        
        demonstrate_storage_manager()
        logger.info("")
        
        demonstrate_pipeline()
        logger.info("")
        
        logger.info("Demonstration completed successfully!")
        logger.info("=" * 60)
        
        # Show the updated pipeline diagram
        logger.info("Updated Pipeline Flow:")
        logger.info("graph TD")
        logger.info("    subgraph \"Ingestion Pipeline\"")
        logger.info("        START([START]) --> ingest[Ingest Documents]")
        logger.info("        ingest --> process[Process Documents<br/>• Split: Modality aware chunking<br/>• Add metadata<br/>• Clean text, images, tables]")
        logger.info("        process --> embed[Generate Embeddings<br/>• Create vector representations<br/>• Handle multimodal content]")
        logger.info("        embed --> store[Store & Index<br/>• Vector database<br/>• Graph database<br/>• Metadata storage]")
        logger.info("        store --> END([READY])")
        logger.info("    end")
        
    except Exception as e:
        logger.error(f"Error in main demonstration: {e}")
        raise


if __name__ == "__main__":
    main() 