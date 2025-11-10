.PHONY: pylint lint format

pylint:
	@echo "Running linter (pylint)..."
	uv run pylint src/ tests/

lint:
	@echo "Running linter (ruff)..."
	uv run ruff check src/ tests/

format:
	@echo "Formatting code with linter (ruff)..."
	uv run ruff format src/ tests/


help:
	@echo "Available commands:"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         - Lint the code with ruff"
	@echo "  make format       - Format the code with ruff"
	@echo "  make pylint       - Lint the code with pylint"
	@echo ""
