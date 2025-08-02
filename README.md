# QuizGenerator

A sophisticated Quiz generation system using LangGraph and LLMs with modular, parallelized architecture and fan-out/fan-in subgraph design.

## 🚀 Features

- **Multi-Agent System**: Coordinated AI agents for quiz generation tasks
- **LangGraph Integration**: Advanced graph-based workflow orchestration
- **Local Model Support**: Optimized for Mistral and other local models
- **RAG Implementation**: Advanced retrieval-augmented generation
- **Embedding & Storage Pipeline**: Vector embeddings and dual database storage
- **Multiple Output Formats**: PDF, DOCX, Markdown support
- **TOGAF Specialization**: Tailored for TOGAF certification preparation
- **Clean Architecture**: SOLID principles and clean code practices

## 📋 Requirements

- Python 3.12+
- Poetry (for dependency management)
- Git

## 🛠️ Installation

### Prerequisites

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd QuizSuite
   ```

### Environment Setup

1. **Install dependencies**:
   ```bash
   poetry install
   ```

2. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

3. **Install development dependencies**:
   ```bash
   poetry install --with dev
   ```

## 🔧 Configuration

### Document Processing Pipeline

The document processing pipeline now includes an advanced embedding and storage phase:

```
START → Ingest Documents → Process Documents → Generate Embeddings → Store & Index → END
```

#### Embedding Phase
- **Vector Embeddings**: Generate embeddings for all document chunks
- **Multimodal Support**: Handle text, images, tables, and figures
- **Parallel Processing**: Efficient batch processing with configurable workers
- **Model Flexibility**: Support for OpenAI and local embedding models

#### Storage Phase
- **Vector Database**: ChromaDB for semantic search and retrieval
- **Graph Database**: JSON-based graph storage for document relationships
- **Metadata Storage**: Rich metadata indexing for advanced queries
- **Dual Storage**: Both vector and graph representations for comprehensive access

#### Configuration Settings

The embedding and storage settings are configured in `config/quiz_config.json`:

```json
{
  "embedding_model": "text-embedding-ada-002",
  "chroma_db_settings": {
    "pdf_persist_directory": "storage/chroma_db_pdf",
    "pdf_collection_name": "pdf_documents",
    "use_existing": true,
    "create_if_not_exists": true
  }
}
```

### Environment Variables

Create a `.env` file in the root directory:

```bash
# API Keys (add your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Model Configuration
DEFAULT_MODEL=mistral
EMBEDDING_MODEL=text-embedding-ada-002

# Database Settings
CHROMA_PERSIST_DIRECTORY=storage/chroma_db_pdf
CHROMA_COLLECTION_NAME=pdf_documents

# Output Settings
OUTPUT_DIRECTORY=backend/output_formats/GenerateQuizes
```

### Quiz Configuration

The main configuration is in `config/quiz_config.json`. Key settings include:

- **Model Configuration**: LLM and embedding model settings
- **Database Settings**: ChromaDB configuration for vector storage
- **Categories**: TOGAF topic categories with document paths
- **Auto Topic Distribution**: Advanced topic coverage settings
- **Cache Settings**: Performance optimization settings

## 🏗️ Project Structure

```
QuizSuite/
├── backend/
│   ├── multi_agents/          # Multi-agent system using LangGraph
│   ├── document_processing/    # PDF, DOCX processing with Unstructured
│   │   ├── processor/         # Document processing components
│   │   │   ├── embedding_manager.py    # Vector embedding generation
│   │   │   └── storage_manager.py      # Database storage management
│   │   ├── state/             # State management for pipeline
│   │   │   └── embed_state.py          # Embedding phase state
│   │   └── node/              # LangGraph nodes
│   ├── report_generation/     # Template-based report generation
│   └── output_formats/        # Multiple output format handlers
├── config/
│   └── quiz_config.json       # Main configuration file
├── storage/
│   ├── TogafD/               # TOGAF documentation PDFs
│   ├── chroma_db_pdf/        # Vector database storage
│   └── graph_database/       # Graph database storage
├── tests/
│   └── integration/
│       └── test_embedding_and_storage.py  # Embedding & storage tests
```
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                 # End-to-end tests
├── output_formats/
│   └── QuestionBanks/        # Generated quiz outputs
├── pyproject.toml            # Poetry configuration
├── .env                      # Environment variables
└── README.md                 # This file
```

## 🚀 Usage

### Basic Quiz Generation

```bash
# Generate quiz using default configuration
poetry run quiz-generator

# Generate quiz with specific configuration
poetry run generate-quiz --config config/quiz_config.json
```

### Development Commands

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=backend --cov-report=html

# Format code
poetry run black backend/

# Sort imports
poetry run isort backend/

# Type checking
poetry run mypy backend/

# Linting
poetry run flake8 backend/
```

## 🧪 Testing

### Test Structure

- **Unit Tests**: `poetry run pytest tests/unit/`
- **Integration Tests**: `poetry run pytest tests/integration/`
- **End-to-End Tests**: `poetry run pytest tests/e2e/`
- **Performance Tests**: `poetry run pytest tests/performance/`

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test categories
poetry run pytest -m unit
poetry run pytest -m integration
poetry run pytest -m e2e

# Run with coverage
poetry run pytest --cov=backend --cov-report=html

# Run in parallel
poetry run pytest -n auto

# Test embedding and storage functionality
poetry run pytest tests/integration/test_embedding_and_storage.py -v
```

### Testing the Embedding and Storage Pipeline

The new embedding and storage functionality includes comprehensive tests:

- **Embedding Manager Tests**: Vector embedding generation and statistics
- **Storage Manager Tests**: Vector and graph database storage
- **Pipeline Integration Tests**: Complete embedding and storage workflow
- **Error Handling Tests**: Robust error handling and recovery

Run the embedding and storage tests specifically:

```bash
poetry run pytest tests/integration/test_embedding_and_storage.py -v
```

## 🔍 Code Quality

### Clean Code Standards

This project follows strict clean code principles:

- **Function Length**: Maximum 20 lines
- **Parameter Count**: Maximum 3 parameters
- **Nesting Depth**: Maximum 3 levels
- **SOLID Principles**: Applied throughout
- **DRY Principle**: No code duplication

### Code Quality Tools

```bash
# Format code
poetry run black backend/

# Sort imports
poetry run isort backend/

# Type checking
poetry run mypy backend/

# Linting
poetry run flake8 backend/

# Pre-commit hooks
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## 📊 Monitoring and Performance

### Performance Metrics

- **Memory Usage**: Monitor in long-running processes
- **Response Time**: Track LLM interaction times
- **Token Usage**: Monitor API costs
- **Cache Hit Rate**: Optimize retrieval performance

### Error Handling

- **API Rate Limiting**: Automatic retry mechanisms
- **Network Timeouts**: Graceful failure handling
- **Invalid Inputs**: Comprehensive validation
- **Source Validation**: Document integrity checks

## 🔐 Security

### Best Practices

- **API Keys**: Stored in `.env` files (never committed)
- **Input Validation**: All user inputs validated
- **Secure File Handling**: Proper file permissions
- **Error Messages**: No sensitive information exposed

### Environment Isolation

- **Poetry Virtual Environments**: Complete isolation
- **Dependency Groups**: Separate dev/prod dependencies
- **Lock File**: Reproducible builds with `poetry.lock`

## 🚀 Deployment

### Production Setup

1. **Install production dependencies**:
   ```bash
   poetry install --only main
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY=your_key
   export ANTHROPIC_API_KEY=your_key
   ```

3. **Run the application**:
   ```bash
   poetry run quiz-generator
   ```

### Docker Support

```dockerfile
FROM python:3.12-slim

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY backend/ ./backend/
COPY config/ ./config/

# Install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --only main

# Run the application
CMD ["poetry", "run", "quiz-generator"]
```

## 🤝 Contributing

### Development Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following clean code principles

3. **Run tests**:
   ```bash
   poetry run pytest
   ```

4. **Format code**:
   ```bash
   poetry run black backend/
   poetry run isort backend/
   ```

5. **Type check**:
   ```bash
   poetry run mypy backend/
   ```

6. **Commit with conventional commits**:
   ```bash
   git commit -m "feat: add new quiz generation feature"
   ```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance tasks

## 📚 Documentation

### Architecture

- **Multi-Agent System**: Coordinated AI agents using LangGraph
- **Document Processing**: PDF/DOCX parsing with Unstructured
- **Vector Database**: ChromaDB for semantic search
- **Output Generation**: Multiple format support

### Key Components

- **LangGraph Workflow**: Fan-out/fan-in parallel processing
- **RAG Implementation**: Advanced retrieval-augmented generation
- **Clean Architecture**: SOLID principles throughout
- **Type Safety**: Comprehensive type hints and validation

## 🐛 Troubleshooting

### Common Issues

1. **Poetry Environment Issues**:
   ```bash
   poetry env remove --all
   poetry install
   ```

2. **Dependency Conflicts**:
   ```bash
   poetry update
   poetry show --tree
   ```

3. **Memory Issues**:
   - Monitor memory usage in long-running processes
   - Implement proper cleanup mechanisms
   - Use streaming for large document processing

4. **API Rate Limiting**:
   - Implement exponential backoff
   - Use request queuing
   - Monitor API usage patterns

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For the LLM integration framework
- **LangGraph**: For the graph-based workflow orchestration
- **ChromaDB**: For the vector database capabilities
- **TOGAF**: For the enterprise architecture framework

## 📞 Support

For support and questions:

1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

**Note**: This project follows strict clean code principles and SOLID design patterns. All contributions must adhere to the established coding standards and testing requirements. 