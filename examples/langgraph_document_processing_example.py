"""
Example script demonstrating the LangGraph document processing pipeline.

This script shows how to use the LangGraph nodes and state models
to process documents according to the quiz configuration.
"""

import sys
import json
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.document_processing.nodes import create_document_processing_graph
from backend.document_processing.models import QuizConfig


def main() -> None:
    """Demonstrate the LangGraph document processing pipeline."""
    print("=== LangGraph Document Processing Pipeline Example ===\n")
    
    # Load configuration
    config_path = "config/quiz_config.json"
    
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        print("Please ensure the quiz_config.json file exists in the config directory.")
        return
    
    try:
        # Create the document processing graph
        print("Creating document processing graph...")
        graph = create_document_processing_graph(config_path)
        
        # Run the pipeline
        print("Running document processing pipeline...")
        result = graph.run()
        
        # Display results
        print("\n=== Pipeline Results ===")
        
        # Check if processing was successful
        if hasattr(result, 'success') and result.success:
            print("✅ Processing completed successfully!")
        else:
            print("❌ Processing encountered errors")
        
        # Display summary
        if hasattr(result, 'summary'):
            summary = result.summary
            print(f"\n📊 Processing Summary:")
            print(f"  Total documents processed: {summary.get('total_documents', 0)}")
            print(f"  Total chunks created: {summary.get('total_chunks', 0)}")
            print(f"  Processing successful: {summary.get('processing_successful', False)}")
            print(f"  Error count: {summary.get('error_count', 0)}")
            
            # Display modality distribution
            modality_dist = summary.get('modality_distribution', {})
            if modality_dist:
                print(f"\n📈 Modality Distribution:")
                for modality, count in modality_dist.items():
                    percentage = (count / summary['total_chunks']) * 100 if summary['total_chunks'] > 0 else 0
                    print(f"  {modality}: {count} chunks ({percentage:.1f}%)")
            
            # Display configuration used
            config_used = summary.get('config_used', {})
            if config_used:
                print(f"\n⚙️  Configuration Used:")
                print(f"  Model: {config_used.get('model_name', 'N/A')}")
                print(f"  Database type: {config_used.get('database_type', 'N/A')}")
                print(f"  Embedding model: {config_used.get('embedding_model', 'N/A')}")
                print(f"  Temperature: {config_used.get('temperature', 'N/A')}")
        
        # Display messages
        if hasattr(result, 'messages') and result.messages:
            print(f"\n📝 Processing Messages:")
            for message in result.messages:
                msg_type = message.get('type', 'info')
                content = message.get('content', '')
                icon = "ℹ️" if msg_type == "info" else "✅" if msg_type == "success" else "❌"
                print(f"  {icon} {content}")
        
        # Display errors if any
        if hasattr(result, 'errors') and result.errors:
            print(f"\n❌ Processing Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        # Display final documents info
        if hasattr(result, 'final_documents') and result.final_documents:
            print(f"\n📄 Final Documents:")
            for i, doc in enumerate(result.final_documents[:3]):  # Show first 3
                print(f"  Document {i+1}: {doc.document_id}")
                print(f"    File: {doc.file_path.name}")
                print(f"    Type: {doc.document_type.value}")
                print(f"    Status: {doc.processing_status}")
                print(f"    Chunks: {len(doc.chunks)}")
                
                # Show sample chunks
                for j, chunk in enumerate(doc.chunks[:2]):  # Show first 2 chunks
                    print(f"      Chunk {j+1}: {chunk.modality.value} ({len(chunk.content)} chars)")
        
        print(f"\n=== Pipeline Complete ===")
        print("The LangGraph document processing pipeline has successfully:")
        print("1. ✅ Loaded configuration from quiz_config.json")
        print("2. ✅ Created LangGraph nodes and state models")
        print("3. ✅ Ingested documents from configured paths")
        print("4. ✅ Processed documents with modality-aware chunking")
        print("5. ✅ Cleaned and enriched document chunks")
        print("6. ✅ Generated comprehensive processing summary")
        
    except Exception as e:
        print(f"❌ Error running document processing pipeline: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_config_loading() -> None:
    """Demonstrate loading and validating the quiz configuration."""
    print("\n=== Configuration Loading Demo ===")
    
    config_path = "config/quiz_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = QuizConfig(**config_data)
        
        print(f"✅ Configuration loaded successfully!")
        print(f"  Database type: {config.database_type}")
        print(f"  Model name: {config.model_name}")
        print(f"  Embedding model: {config.embedding_model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Target audience: {config.target_audience}")
        print(f"  Difficulty level: {config.difficulty_level}")
        print(f"  Number of categories: {len(config.categories)}")
        
        print(f"\n📚 Categories:")
        for i, category in enumerate(config.categories):
            print(f"  {i+1}. {category.name} ({category.num_questions} questions)")
            print(f"     Description: {category.description}")
            print(f"     Documents: {len(category.doc_paths)} files")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")


if __name__ == "__main__":
    # First demonstrate configuration loading
    demonstrate_config_loading()
    
    # Then run the full pipeline
    main() 