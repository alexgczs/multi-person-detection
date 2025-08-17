.PHONY: help install install-dev test test-cov lint clean build

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (flake8)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest

test-cov:
	pytest --cov=src --cov-report=term-missing --cov-report=html

# Code quality
lint:
	flake8 src/ tests/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Build
build:
	python -m build
