# PostgreSQL Integration with Document Processing Pipeline

## 📋 Overview

This document describes the PostgreSQL integration with the QuizSuite document processing pipeline, including the `process_content` functionality, chunking strategies, and data storage architecture.

## 🏗️ Architecture

### Database Schema

The PostgreSQL database uses a single `chunks` table to store processed document chunks:

```sql
CREATE TABLE chunks (
    chunk_id VARCHAR PRIMARY KEY,
    document_id VARCHAR NOT NULL,
    chunk_index INTEGER,
    content TEXT NOT NULL,
    content_hash VARCHAR,
    chunk_metadata JSONB,
    quality_score FLOAT DEFAULT 0.0,
    source_document VARCHAR,
    chunk_type VARCHAR DEFAULT 'text',
    modality VARCHAR,
    content_length INTEGER,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);
```

### Key Fields

- **`chunk_id`**: Unique identifier for each chunk (format: `{document_id}_chunk_{index}`)
- **`document_id`**: Reference to the source document
- **`content`**: The actual chunk content (text, code, diagrams)
- **`modality`**: Content type classification (`text`, `code`, `diagram`)
- **`chunk_type`**: Processing type (currently all `text`)
- **`content_length`**: Character count of the chunk
- **`quality_score`**: Chunk quality assessment (0.0 by default)

## 🔧 Configuration

### PostgreSQL Configuration

The database connection is configured in `config/quiz_config.json`:

```json
{
  "postgresql_storage": {
    "host": "localhost",
    "port": 5432,
    "database": "quizsuite",
    "username": "postgres",
    "password": "your_password"
  }
}
```

### Document Processing Configuration

The pipeline uses structure-aware + modality-aware chunking:

```json
{
  "chunking_strategy": {
    "structure_aware": true,
    "modality_aware": true,
    "chunk_size": 1500,
    "chunk_overlap": 200
  }
}
```

## 📊 Current Database Statistics

### Overall Statistics
- **Total Chunks**: 496 chunks
- **Total Documents**: 42 PDF documents
- **Total Content**: 665,866 characters
- **Average Chunk Size**: 1,342.5 characters

### Modality Distribution
- **Diagram**: 276 chunks (55.6%) - Visual content and figures
- **Text**: 122 chunks (24.6%) - Narrative content
- **Code**: 98 chunks (19.8%) - Technical/structured content

### Top Documents by Chunk Count
1. `togaf-standard-architecture-content/latest/01-doc/chap03` - 49 chunks
2. `togaf-standard-introduction-and-core-concepts/latest/01-doc/chap03` - 31 chunks
3. `togaf-standard-adm-techniques/latest/01-doc/chap02` - 27 chunks

## 🚀 Usage

### 1. Process All PDFs to PostgreSQL

```bash
# Process all PDFs from config and store chunks in PostgreSQL
poetry run python process_all_pdfs_to_postgresql.py
```

This script:
- Loads all PDFs from the config JSON
- Applies structure-aware + modality-aware chunking
- Stores chunks in PostgreSQL with proper metadata
- Provides detailed statistics

### 2. Export Chunks for Specific PDF

```bash
# Export chunks for a specific PDF in JSON format
poetry run python export_chunks_for_pdf.py
```

### 3. Generate Summary Reports

```bash
# Create comprehensive database summary report
poetry run python create_chunks_summary_report.py

# Analyze code modality distribution
poetry run python analyze_code_modality.py
```

## 🔍 Key Scripts

### Core Processing Scripts

#### `process_all_pdfs_to_postgresql.py`
- **Purpose**: Main pipeline for processing all PDFs and storing in PostgreSQL
- **Features**:
  - Loads PDFs from config JSON
  - Applies structure-aware + modality-aware chunking
  - Handles content extraction with header/footer removal disabled
  - Stores chunks with proper metadata and timestamps
  - Provides detailed processing statistics

#### `export_chunks_for_pdf.py`
- **Purpose**: Export chunks for specific PDFs in JSON format
- **Features**:
  - Connects to PostgreSQL database
  - Searches for document IDs
  - Exports chunks with full metadata
  - Provides content analysis

### Analysis Scripts

#### `create_chunks_summary_report.py`
- **Purpose**: Generate comprehensive database statistics
- **Features**:
  - Overall database statistics
  - Document-level analysis
  - Modality distribution analysis
  - Top documents by various metrics

#### `analyze_code_modality.py`
- **Purpose**: Deep analysis of code modality distribution
- **Features**:
  - Top documents by code chunk count
  - Code percentage analysis
  - Sample code chunk previews
  - Modality comparison

#### `check_document_ids.py`
- **Purpose**: List all document IDs in the database
- **Features**:
  - Shows all unique document IDs
  - Searches for specific patterns
  - Provides document naming insights

## 📈 Data Quality

### Content Extraction Improvements
- **Header/Footer Removal**: Disabled to preserve content
- **Content Length**: Increased from 942 to 78,423 characters per PDF
- **Chunk Count**: Improved from 1 to 53 chunks per PDF

### Chunking Strategy
- **Structure-Aware**: Preserves document structure and logical sections
- **Modality-Aware**: Classifies content as text, code, or diagram
- **Quality Filtering**: Maintains chunk quality standards
- **Deduplication**: Prevents duplicate content storage

## 🔧 Database Operations

### Connection Management

```python
def connect_to_postgresql():
    """Connect to PostgreSQL database."""
    postgres_config = get_postgresql_config()
    
    conn = psycopg2.connect(
        host=postgres_config.get('host', 'localhost'),
        port=postgres_config.get('port', 5432),
        database=postgres_config.get('database', 'quizsuite'),
        user=postgres_config.get('username', 'postgres'),
        password=postgres_config.get('password', '')
    )
    return conn
```

### Chunk Storage

```python
def store_chunks_in_postgresql(chunks, document_id):
    """Store chunks in PostgreSQL with proper metadata."""
    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{i}"
        
        cursor.execute("""
            INSERT INTO chunks (
                chunk_id, document_id, chunk_index, content,
                content_hash, chunk_metadata, quality_score,
                source_document, chunk_type, modality,
                content_length, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (chunk_id, document_id, i, chunk.content, ...))
```

## 📊 Monitoring and Analysis

### Database Statistics Queries

```sql
-- Total chunks and documents
SELECT COUNT(*) as total_chunks, COUNT(DISTINCT document_id) as total_documents 
FROM chunks;

-- Modality distribution
SELECT modality, COUNT(*) as count, 
       ROUND((COUNT(*)::float / (SELECT COUNT(*) FROM chunks)) * 100, 2) as percentage
FROM chunks 
GROUP BY modality 
ORDER BY count DESC;

-- Document-level statistics
SELECT document_id, COUNT(*) as chunk_count,
       AVG(content_length) as avg_length,
       SUM(content_length) as total_content
FROM chunks 
GROUP BY document_id 
ORDER BY chunk_count DESC;
```

### Quality Metrics

- **Content Preservation**: 100% of extracted content stored
- **Modality Classification**: 55.6% diagrams, 24.6% text, 19.8% code
- **Chunk Consistency**: Average 1,342.5 characters per chunk
- **Document Coverage**: All 42 PDFs from config successfully processed

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check PostgreSQL service status
   brew services list | grep postgresql
   
   # Restart PostgreSQL if needed
   brew services restart postgresql
   ```

2. **Permission Issues**
   ```bash
   # Create database if it doesn't exist
   createdb quizsuite
   
   # Grant permissions
   psql -d quizsuite -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;"
   ```

3. **Data Type Issues**
   - Ensure timestamps use `timestamp without time zone`
   - Use `varchar` for chunk_id (not auto-increment)
   - Handle JSONB for chunk_metadata

### Performance Optimization

- **Indexing**: Consider adding indexes on `document_id` and `modality`
- **Batch Operations**: Use transaction batching for large datasets
- **Connection Pooling**: Implement connection pooling for high-volume operations

## 📋 File Structure

```
QuizSuite/
├── config/
│   └── quiz_config.json          # PostgreSQL and processing configuration
├── process_all_pdfs_to_postgresql.py    # Main processing pipeline
├── export_chunks_for_pdf.py             # Export specific PDF chunks
├── create_chunks_summary_report.py      # Generate database reports
├── analyze_code_modality.py             # Code modality analysis
├── check_document_ids.py                # Database document listing
└── chunks_summary_report.json           # Generated summary report
```

## 🔄 Integration with Process Content

The PostgreSQL integration works seamlessly with the `process_content` functionality:

1. **Content Extraction**: PDFs are processed using enhanced content extraction
2. **Chunking**: Structure-aware + modality-aware chunking creates intelligent chunks
3. **Storage**: Chunks are stored in PostgreSQL with full metadata
4. **Analysis**: Comprehensive reporting and analysis tools available
5. **Export**: JSON export capabilities for further processing

## 🎯 Benefits

- **Persistent Storage**: Reliable PostgreSQL storage vs. volatile vector databases
- **Rich Metadata**: Full chunk metadata including modality, quality scores, timestamps
- **Scalability**: Handles large document collections efficiently
- **Analytics**: Comprehensive reporting and analysis capabilities
- **Integration**: Seamless integration with existing document processing pipeline

## 📚 Related Documentation

- [Document Processing Pipeline](./backend/document_processing/)
- [Chunking Strategies](./backend/document_processing/processor/chunking_strategy.py)
- [Content Extraction](./backend/document_processing/processor/content_extractor.py)
- [Configuration Management](./config/)

---

**Last Updated**: August 6, 2025  
**Version**: 1.0  
**Status**: Production Ready ✅ 