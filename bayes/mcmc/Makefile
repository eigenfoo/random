.PHONY: help venv format
.DEFAULT_GOAL = help

PYTHON = python3
SHELL = bash

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

venv:  # Set up a virtual environment
	( \
	rm -rf venv; \
	${PYTHON} -m venv venv; \
	source venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt; \
	deactivate; \
	)
