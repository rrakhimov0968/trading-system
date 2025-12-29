.PHONY: help install test test-cov test-unit test-integration lint format clean run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install black mypy pylint

test: ## Run all tests
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest -m unit

test-integration: ## Run integration tests only
	pytest -m integration

test-fast: ## Run fast tests (exclude slow and API tests)
	pytest -m "not slow and not api"

lint: ## Run linters
	pylint agents/ config/ models/ utils/
	mypy agents/ config/ models/ utils/

format: ## Format code with black
	black agents/ config/ models/ utils/ tests/ main.py

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build

run: ## Run the main application
	python main.py

run-tests: test ## Alias for test

ci: lint test ## Run linting and tests (for CI)

