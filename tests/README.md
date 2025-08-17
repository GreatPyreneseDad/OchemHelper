# OChem Helper Test Suite

Comprehensive test suite for the OChem Helper molecular discovery platform.

## Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_vae.py      # VAE model tests
│   ├── test_api.py      # API endpoint tests
│   └── test_mcp.py      # MCP server tests
├── integration/         # Integration tests
│   └── test_integration.py
├── run_tests.py        # Test runner script
└── requirements-test.txt # Test dependencies
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r tests/requirements-test.txt
```

### Run All Tests

```bash
python tests/run_tests.py
```

### Run Specific Test Types

```bash
# Unit tests only
python tests/run_tests.py --unit

# Integration tests only
python tests/run_tests.py --integration

# With coverage report
python tests/run_tests.py --coverage
```

### Run Specific Tests

```bash
# Run a specific test file
python tests/run_tests.py unit/test_vae.py

# Run a specific test class
python tests/run_tests.py unit/test_vae.py::TestSMILESTokenizer

# Run a specific test method
python tests/run_tests.py unit/test_vae.py::TestSMILESTokenizer::test_encode_basic
```

## Test Coverage

### Unit Tests

- **VAE Tests (`test_vae.py`)**
  - SMILES tokenizer functionality
  - VAE model components
  - Training utilities
  - Generation and reconstruction

- **API Tests (`test_api.py`)**
  - All REST endpoints
  - Request/response validation
  - Error handling
  - WebSocket connections

- **MCP Tests (`test_mcp.py`)**
  - MCP server initialization
  - Tool registration and execution
  - Property prediction tool
  - Synthesis suggestion tool
  - Structure optimization tool
  - Reaction prediction tool

### Integration Tests

- **Data Pipeline**: Download and processing
- **Model Training**: End-to-end training
- **API Integration**: Full API workflow
- **MCP Integration**: Complete MCP workflow
- **CLI Integration**: Command-line tools
- **Container Integration**: Podman builds

## Writing New Tests

### Unit Test Template

```python
import unittest
from unittest.mock import Mock, patch

class TestMyComponent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.component = MyComponent()
    
    def test_functionality(self):
        """Test specific functionality."""
        result = self.component.method()
        self.assertEqual(result, expected)
    
    @patch('module.external_dependency')
    def test_with_mock(self, mock_dep):
        """Test with mocked dependencies."""
        mock_dep.return_value = 'mocked'
        result = self.component.method()
        self.assertEqual(result, 'expected')
```

### Async Test Template

```python
import unittest
import asyncio

class TestAsyncComponent(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()
    
    async def test_async_method(self):
        result = await async_function()
        self.assertEqual(result, expected)
    
    def test_run_async(self):
        self.loop.run_until_complete(self.test_async_method())
```

## Test Guidelines

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external dependencies
3. **Coverage**: Aim for >80% code coverage
4. **Naming**: Use descriptive test names
5. **Assertions**: Use specific assertions
6. **Setup/Teardown**: Clean up resources properly

## Continuous Integration

Tests are automatically run on:
- Push to main branch
- Pull requests
- Nightly builds

See `.github/workflows/test.yml` for CI configuration.

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the project root
   - Check Python path includes src/

2. **Missing Dependencies**
   - Install test requirements
   - Install main requirements.txt

3. **API Tests Failing**
   - Check if API server is running
   - Verify port 8001 is available

4. **Integration Tests Slow**
   - Use --unit flag for faster feedback
   - Mock external services when possible

### Debug Mode

Run tests with verbose output:
```bash
python tests/run_tests.py -v 2
```

Stop on first failure:
```bash
python tests/run_tests.py --failfast
```

## Performance Testing

For performance-critical components:

```python
import pytest

@pytest.mark.benchmark
def test_performance(benchmark):
    result = benchmark(function_to_test, arg1, arg2)
    assert result == expected
```

Run benchmarks:
```bash
pytest tests/unit/test_performance.py --benchmark-only
```