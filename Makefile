.PHONY: test test-cov pylint lint format etl

test:
	uv run pytest tests/ -v

test-cov:
	@echo "Test execution with coverage report"
	uv run pytest --cov=src --cov-report=html tests/

pylint:
	@echo "Running linter (pylint)..."
	uv run pylint src/ tests/

lint:
	@echo "Running linter (ruff)..."
	uv run ruff check src/ tests/

format:
	@echo "Formatting code with linter (ruff)..."
	uv run ruff format src/ tests/

etl:
	@echo "Download dataset..."
	uv run python -m src.etl.download_dataset


help:
	@echo "Available commands:"
	@echo "Execution:"
	@echo "  make etl          - Download and extract images dataset"
	@echo ""
	@echo "Tests:"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Lint the code with ruff"
	@echo "  make format       - Format the code with ruff"
	@echo "  make pylint       - Lint the code with pylint"
	@echo ""
