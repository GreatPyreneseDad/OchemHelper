# Contributing to OChem Helper

We welcome contributions from the community! This guide will help you get started.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/ochem-helper.git
   cd ochem-helper
   ```

3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. Set up your development environment:
   ```bash
   make install
   ```

## Development Workflow

### Code Style

We use:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run all formatters and linters:
```bash
make format
make lint
```

### Testing

Write tests for all new functionality:
```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_molecular_graph.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:
```bash
pre-commit install
pre-commit run --all-files
```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update documentation for any API changes
3. Add tests for new functionality
4. Ensure all tests pass and coverage remains high
5. Update the CHANGELOG.md
6. Submit a pull request

## Code of Conduct

Please be respectful and constructive in all interactions.

## Questions?

Feel free to open an issue for any questions!