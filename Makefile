.PHONY: help install test lint format clean podman-build podman-run

help:
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  clean         Clean build artifacts"
	@echo "  podman-build  Build container image with Podman"
	@echo "  podman-run    Run container with Podman"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

podman-build:
	podman build -t ochem-helper:latest -f containers/Containerfile .

podman-run:
	podman run -it --rm -p 8000:8000 ochem-helper:latest

train:
	python scripts/train_model.py --config configs/default.yaml

generate:
	python scripts/generate_molecules.py --model models/pretrained/latest.pt

api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

download-data:
	python scripts/download_data.py --all

setup-gpu:
	conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
	pip install torch-geometric
	pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html