.PHONY: test test-cov pylint lint format etl index-images preprocess-images help

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

index-images:
	@echo "Index images ad labels..."
	uv run python -m src.etl.index_images_and_labels

preprocess-images:
	@echo "Preprocess split images..."
	uv run python -m src.etl.preprocess_images_split

help:
	@echo "Available commands:"
	@echo "Execution:"
	@echo "  make etl               - Download and extract images dataset"
	@echo "  make index-images      - Index images in the raw data directory"
	@echo "  make preprocess-images - Preprocess indexed images (resize, normalize)"
	@echo ""
	@echo "Tests:"
	@echo "  make test              - Run tests"
	@echo "  make test-cov          - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Lint the code with ruff"
	@echo "  make format            - Format the code with ruff"
	@echo "  make pylint            - Lint the code with pylint"
	@echo ""
