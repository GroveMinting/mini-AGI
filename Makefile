# Hardening mirror of volotat/mini-AGI. See UPSTREAM.md.
VENV ?= $(HOME)/.venvs/minagi-hardening
PY   := $(VENV)/bin/python
ARGS ?=

.PHONY: help setup setup-corpora test sync patches run serve clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## create the venv and install deps (CPU torch, enough for the tests)
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install -q -U pip
	$(VENV)/bin/python -m pip install -q -r requirements.txt
	$(VENV)/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
	@echo "For GPU training, install the CUDA/ROCm/torch build into $(VENV)."

setup-corpora: ## extra deps for the corpus builders (Hugging Face, chess)
	$(VENV)/bin/python -m pip install -q -r requirements-corpora.txt

test: ## run the CPU test suite
	$(PY) -m unittest discover -s tests

sync: ## fetch upstream, rebase the patchset, re-export patches, run tests
	scripts/sync-upstream.sh

patches: ## re-export the patch series from the current branch
	rm -f patches/*.patch
	git format-patch --no-signature -o patches upstream..HEAD
	ls -1 patches

run: ## read a corpus into weights/ (override with ARGS=...)
	$(PY) train.py read data/train --save --weights-dir weights \
		--held-out data/val --sample-every 10 $(ARGS)

serve: ## serve the local web UI on http://127.0.0.1:8080
	$(PY) serve.py --port 8080

clean: ## remove caches
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
