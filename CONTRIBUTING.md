# Contributing to H4wkQuant

Thank you for your interest in contributing to H4wkQuant! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/H4wkQuant.git`
3. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
# Copy environment file
cp .env.example .env

# Start dependencies (Redis, PostgreSQL)
make up

# Run tests
make test

# Run backtest
make backtest
```

## How to Contribute

### Reporting Bugs
- Use GitHub Issues
- Describe the bug clearly
- Include steps to reproduce
- Add logs if applicable

### Suggesting Features
- Open an issue with "Feature Request" label
- Describe the feature and its benefits
- Discuss implementation approach

### Pull Requests
1. Ensure tests pass
2. Update documentation if needed
3. Follow existing code style
4. Write clear commit messages

## Code Style

- Python: PEP 8
- Type hints encouraged
- Docstrings for public functions
- Comments for complex logic

## Contact

- GitHub Issues: [github.com/H4wkCode/H4wkQuant/issues](https://github.com/H4wkCode/H4wkQuant/issues)
- Discussions: [github.com/H4wkCode/H4wkQuant/discussions](https://github.com/H4wkCode/H4wkQuant/discussions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
