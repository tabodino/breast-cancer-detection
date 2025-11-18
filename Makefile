.PHONY: test test-cov pylint lint format etl index-images preprocess-images train evaluate serve mlflow-ui streamlit run-all help

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

train:
	@echo "Training models with MLflow..."
	uv run python -m src.models.training

evaluate:
	@echo "Evaluating models..."
	uv run python -m src.evaluation.metrics

serve:
	@echo "Starting FastAPI server..."
	uv run uvicorn src.api.main:app --reload

mlflow-ui:
	@echo "Launch MLflow UI..."
	uv run mlflow ui

streamlit:
	@echo "Starting Streamlit UI..."
	uv run streamlit run src/streamlit/app.py --server.port=8501

run-all:
	@echo "Starting FastAPI + Streamlit  + MLFlow services..."
	uv run streamlit run src/streamlit/app.py --server.port=8501 &
	uv run mlflow ui &
	uv run uvicorn src.api.main:app --reload

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

help:
	@echo "Available commands:"
	@echo "Execution:"
	@echo "  make etl               - Download and extract images dataset"
	@echo "  make index-images      - Index images in the raw data directory"
	@echo "  make preprocess-images - Preprocess indexed images (resize, normalize)"
	@echo ""
	@echo "Training and Evaluation:"
	@echo "  make train             - Train models with MLflow"
	@echo "  make evaluate          - Evaluate trained models"
	@echo ""
	@echo "Serving:"
	@echo "  make serve             - Start FastAPI server"
	@echo "  make mlflow-ui         - Launch MLflow UI"
	@echo "  make streamlit         - Run Streamlit UI"
	@echo "  make run-all           - Run all services together"
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
	@echo "Docker:"
	@echo "  make docker-up    		- Start Docker services"
	@echo "  make docker-down  		- Stop Docker services"
	@echo "  make docker-logs  		- See Docker logs"
	@echo "  make docker-build 		- Build Docker images"
