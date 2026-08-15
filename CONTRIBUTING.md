# Contributing to Souprise

Thank you for your interest in contributing to Souprise! We welcome contributions from everyone.

## Ways to Contribute

- **Bug Reports**: Open an issue on GitHub with a clear description and steps to reproduce.
- **Feature Requests**: Open an issue to discuss new features before implementing.
- **Code Contributions**: Submit a pull request with your changes.
- **Documentation**: Improve docs, add examples, or fix typos.
- **Tests**: Add test cases to improve coverage.

## Development Setup

### 1. Fork the Repository

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/your-username/souprise.git
cd souprise
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Install Pre-commit Hooks (Optional)

```bash
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=souprise --cov-report=term-missing

# Run a specific test
pytest tests/test_pipeline.py
```

## Code Style

Souprise uses [Ruff](https://github.com/astral-sh/ruff) for linting:

```bash
# Check code style
ruff check souprise/

# Fix automatically fixable issues
ruff check --fix souprise/
```

### Style Guidelines

- **Line Length**: 100 characters maximum
- **Imports**: Grouped by standard library, third-party, local (with blank lines between)
- **Type Hints**: Use Python type hints for all public functions and methods
- **Docstrings**: Use Google-style docstrings for all public functions
- **Naming**: Use snake_case for variables/functions, CamelCase for classes
- **Quotes**: Use double quotes for strings, single quotes for docstrings

## Pull Request Guidelines

1. **Create a Feature Branch**: Use a descriptive name (e.g., `feat/add-postgres-support`)
2. **Write Tests**: Add tests for new functionality
3. **Update Docs**: Update documentation for any API changes
4. **Keep Commits Atomic**: Each commit should be a single logical change
5. **Write Good Commit Messages**: Use the [Conventional Commits](https://www.conventionalcommits.org/) format

### Commit Message Format

```
type(scope): subject

body

footer
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(pipeline): add support for batch queries

- Add batch_query method to SoupriseRAG class
- Support processing multiple queries in parallel
- Add benchmark for batch performance

Fixes #123
```

## License

By contributing to Souprise, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Code of Conduct

We expect all contributors to follow our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, inclusive, and professional.

## Reporting Issues

When reporting issues, please include:

- **Python version**: `python --version`
- **Operating System**: macOS/Linux/Windows, version
- **Souprise version**: `souprise version`
- **Steps to reproduce**: Clear, minimal steps
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full traceback if applicable

## Design Decisions

Major design decisions are documented in the README architecture section (if it exists). For new features, please open an issue to discuss the design before implementing.

## Maintainers

- [Michael Kupermann](https://github.com/mkupermann)

## Acknowledgments

Thank you to all contributors who have helped make Souprise better!
