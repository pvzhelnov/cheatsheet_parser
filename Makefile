# Makefile for setting up Python project using Conda + Poetry + requirements.txt

PYTHON_VERSION = 3.12.9  # FYI only - hardcoded in config files below
PIP_VERSION = 25.1       # FYI only - hardcoded in config files below
POETRY_VERSION = 2.1.3   # FYI only - hardcoded in config files below
CONDA_ENV_NAME = cheatsheet_parser

# Specifies runtime and environment
CONDA_CONFIG_FILE = ./environment.yml

# Specifies Python packaging
POETRY_CONFIG_FILE = ./pyproject.toml

# Lists Python dependencies
PIP_REQUIREMENTS_FILE = ./requirements.txt

.PHONY: install conda poetry uninstall

install: conda poetry

conda:
	@echo "📦 Creating Conda environment from file: $(CONDA_CONFIG_FILE)"
	@conda env create -f $(CONDA_CONFIG_FILE) -n $(CONDA_ENV_NAME)
	@echo "✅ Conda environment created."

poetry:
	@echo "🛠️ Configuring Poetry to use Conda env"
	@conda run -n $(CONDA_ENV_NAME) poetry config virtualenvs.create false
	@echo "🔄 Adding dependencies from $(PIP_REQUIREMENTS_FILE) via Poetry"
	@conda run -n $(CONDA_ENV_NAME) poetry add $(shell cat $(PIP_REQUIREMENTS_FILE))
	@echo "🚀 Installing dependencies via Poetry"
	@conda run -n $(CONDA_ENV_NAME) poetry install
	@echo "✅ Environment ready!"

uninstall:
	@echo "🧹 Removing Conda environment $(CONDA_ENV_NAME)"
	@conda env remove -n $(CONDA_ENV_NAME)