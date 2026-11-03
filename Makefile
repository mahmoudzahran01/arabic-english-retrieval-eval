PY ?= python

.PHONY: install install-train test lint format report chart

install:            ## CPU-only: tests, tables, chart
	$(PY) -m pip install -e ".[dev]"

install-train:      ## GPU stages: torch + sentence-transformers (add [rageval] for the Ragas part)
	$(PY) -m pip install -e ".[dev,train]"

test:               ## unit tests + checks that the committed results regenerate the committed tables
	$(PY) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

report:             ## tables.md -> docs/retrieval/RESULTS.md -> docs/img/*.png, all from results/
	$(PY) -m arretrieval.retrieval.make_report
	$(PY) -m arretrieval.retrieval.build_results
	$(PY) scripts/make_chart.py

chart:
	$(PY) scripts/make_chart.py
