# Spryx HTTP

A Python HTTP client library for Spryx services.

## Installation

```bash
pip install spryx-http
```

## Development

### Requirements

- Python 3.11 or 3.12
- Poetry for dependency management
- Tox for testing

### Setting up the development environment

```bash
# Install dependencies
poetry install

# Install tox
pip install tox
```

### Running tests

```bash
# Run all tests
tox

# Run tests for specific Python version
tox -e py311
tox -e py312

# Run linting checks
tox -e lint

# Run type checking
tox -e typecheck

# Run formatting
tox -e format
```

## License

See the LICENSE file for details. 