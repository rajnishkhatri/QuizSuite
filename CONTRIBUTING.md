# Contributing to QuizSuite

Thank you for your interest in contributing to QuizSuite! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome contributions in the following areas:

- **🐛 Bug Reports**: Help us identify and fix issues
- **✨ Feature Requests**: Suggest new features and improvements
- **📚 Documentation**: Improve or add documentation
- **🧪 Tests**: Add or improve test coverage
- **🔧 Code Improvements**: Refactor, optimize, or enhance existing code
- **🎨 UI/UX**: Improve user interface and experience

### Before You Start

1. **Check existing issues**: Search for similar issues or feature requests
2. **Read the documentation**: Familiarize yourself with the project structure
3. **Set up development environment**: Follow the installation guide in README.md

## 🛠️ Development Setup

### Prerequisites

- Python 3.12+
- Poetry
- Git

### Local Development

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/quiz-suite.git
   cd quiz-suite
   ```

3. **Set up the development environment**:
   ```bash
   poetry install --with dev
   poetry shell
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

5. **Make your changes**

6. **Run tests**:
   ```bash
   poetry run pytest
   ```

7. **Format and lint code**:
   ```bash
   poetry run black .
   poetry run isort .
   poetry run flake8
   poetry run mypy .
   ```

8. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

9. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

10. **Create a pull request**

## 📝 Code Style Guidelines

### Python Code Style

We follow PEP 8 and use automated tools for code formatting:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat: add support for custom quiz templates
fix: resolve embedding generation error
docs: update installation instructions
test: add unit tests for document processing
```

### Code Quality Standards

- **Functions**: Keep under 20 lines, max 3 parameters
- **Classes**: Follow Single Responsibility Principle
- **Documentation**: Add docstrings to all functions and classes
- **Type Hints**: Use type hints for all function parameters and return values
- **Error Handling**: Implement proper exception handling
- **Testing**: Maintain good test coverage

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test types
poetry run pytest tests/unit/
poetry run pytest tests/integration/
poetry run pytest tests/e2e/

# Run with coverage
poetry run pytest --cov=backend --cov-report=html

# Run performance tests
poetry run pytest -m slow
```

### Writing Tests

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Test Naming**: Use descriptive test names
- **Test Organization**: Group related tests in classes

**Example**:
```python
import pytest
from backend.document_processing.processor import DocumentProcessor

class TestDocumentProcessor:
    def test_process_single_document(self):
        """Test processing a single document."""
        processor = DocumentProcessor()
        result = processor.process("test.pdf")
        assert result is not None
        assert result.document_id == "test.pdf"

    def test_process_invalid_document(self):
        """Test processing an invalid document."""
        processor = DocumentProcessor()
        with pytest.raises(ValueError):
            processor.process("invalid.txt")
```

## 📚 Documentation Standards

### Code Documentation

- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Include type hints for all functions
- **Examples**: Provide usage examples in docstrings

**Example**:
```python
def process_document(file_path: Path) -> ProcessedDocument:
    """Process a document and extract its content.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        ProcessedDocument: The processed document with extracted content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
        
    Example:
        >>> doc = process_document(Path("document.pdf"))
        >>> print(doc.document_id)
        'document.pdf'
    """
```

### Documentation Structure

- **README.md**: Project overview and quick start
- **docs/architecture/**: System architecture documentation
- **docs/api/**: API documentation
- **docs/user_guides/**: User guides and tutorials

## 🔍 Pull Request Process

### Before Submitting

1. **Ensure tests pass**: All tests must pass
2. **Check code quality**: Run linting and type checking
3. **Update documentation**: Add or update relevant documentation
4. **Test your changes**: Verify your changes work as expected

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Test addition/update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Commit messages follow conventional format

## Additional Notes
Any additional information or context
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: Maintainers review the code
3. **Feedback**: Address any feedback or requested changes
4. **Merge**: Once approved, the PR is merged

## 🐛 Bug Reports

### Before Reporting

1. **Check existing issues**: Search for similar issues
2. **Reproduce the issue**: Ensure you can reproduce it consistently
3. **Gather information**: Collect relevant logs and error messages

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., macOS 13.0]
- Python Version: [e.g., 3.12.0]
- Poetry Version: [e.g., 1.7.0]

## Additional Information
- Error messages
- Logs
- Screenshots (if applicable)
```

## ✨ Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature is needed and how it would be used

## Proposed Implementation
Optional: Suggestions for implementation

## Alternatives Considered
Optional: Other approaches you've considered

## Additional Information
Any other relevant information
```

## 🏷️ Issue Labels

We use the following labels to categorize issues:

- **bug**: Something isn't working
- **enhancement**: New feature or request
- **documentation**: Improvements or additions to documentation
- **good first issue**: Good for newcomers
- **help wanted**: Extra attention is needed
- **priority: high**: High priority issue
- **priority: medium**: Medium priority issue
- **priority: low**: Low priority issue

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private or sensitive matters

### Before Asking for Help

1. **Check documentation**: Read the README and docs
2. **Search issues**: Look for similar questions
3. **Provide context**: Include relevant information
4. **Be specific**: Describe your issue clearly

## 🙏 Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **CHANGELOG.md**: Credit for significant contributions
- **Release notes**: Acknowledgment in releases

## 📄 License

By contributing to QuizSuite, you agree that your contributions will be licensed under the MIT License.

## 🚀 Quick Reference

### Common Commands

```bash
# Install dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run isort .

# Lint code
poetry run flake8
poetry run mypy .

# Run specific test
poetry run pytest tests/unit/test_document_processor.py

# Generate coverage report
poetry run pytest --cov=backend --cov-report=html
```

### Development Workflow

1. Fork → Clone → Branch → Code → Test → Commit → Push → PR
2. Keep branches focused and small
3. Write clear commit messages
4. Add tests for new features
5. Update documentation as needed

Thank you for contributing to QuizSuite! 🎉 