#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mibici
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
DATA_ARGS ?=

## Download raw MiBici data
.PHONY: data
data: 
	conda run -n $(PROJECT_NAME) python -m mibici.dataset $(DATA_ARGS)

PROFILE_ARGS ?=

## Profile raw MiBici data (read-only) into reports/profiling
.PHONY: profile
profile:
	conda run -n $(PROJECT_NAME) python -m mibici.profiling $(PROFILE_ARGS)

INTERIM_ARGS ?=

## Build data/interim (trips + stations) from data/raw and write reports/interim
.PHONY: interim
interim:
	conda run --no-capture-output -n $(PROJECT_NAME) python -m mibici.interim $(INTERIM_ARGS)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)



#################################################################################
# Register the project Jupyter kernel                                           #
#################################################################################

## Register the project Jupyter kernel
.PHONY: kernel
kernel:
	conda run -n $(PROJECT_NAME) python -m ipykernel install \
		--user \
		--name $(PROJECT_NAME) \
		--display-name "Python ($(PROJECT_NAME))"


## Full project setup
.PHONY: setup
setup:
	@if conda env list | awk '{print $$1}' | grep -qx "$(PROJECT_NAME)"; then \
		echo ">>> Environment exists. Updating..."; \
		$(MAKE) requirements; \
	else \
		echo ">>> Creating environment..."; \
		$(MAKE) create_environment; \
	fi
	$(MAKE) kernel
	@echo ">>> MiBici environment ready."


#################################################################################
# Set up the MLflow structure and start the server                              #
#################################################################################

MLFLOW_DIR = $(CURDIR)/mlflow
MLFLOW_PORT ?= 5000

## Start local MLflow Tracking Server (SQLite backend + local artifacts)
.PHONY: mlflow
mlflow:
	mkdir -p $(MLFLOW_DIR)/backend $(MLFLOW_DIR)/artifacts
	conda run --no-capture-output -n $(PROJECT_NAME) mlflow server \
		--backend-store-uri sqlite:///$(MLFLOW_DIR)/backend/mlflow.db \
		--default-artifact-root $(MLFLOW_DIR)/artifacts \
		--host 127.0.0.1 --port $(MLFLOW_PORT)
