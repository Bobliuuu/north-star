# Northstar one-command targets. CPU only; nothing here spends GPU money.
#   make setup   create .venv with the pinned versions from SETUP.md
#   make test    all CPU unit tests (each suite in its own pytest process: tests/ and src/approx/tests both define `fixtures`)
#   make check   correctness runner: exactness of the verifier path / fused-vs-two-pass reference on CPU (subset of `test`)
#   make bench   PRINTS the exact modal --detach commands (GPU, costs money); does NOT run them
#   make plots   regenerate every plot from raw results/*.jsonl into results/plots/
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
TEST_DIRS := tests src/approx src/kvm src/opt src/selfimprove src/tage
CHECK_DIRS := tests src/approx src/kvm

.PHONY: setup test check bench plots all

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	@set -e; for d in $(TEST_DIRS); do echo "== pytest $$d"; $(PY) -m pytest -q $$d; done

check:
	@set -e; for d in $(CHECK_DIRS); do echo "== correctness $$d"; $(PY) -m pytest -q $$d; done
	@echo "CPU correctness only (tiny-fixture RefNorth vs HF, spec == plain greedy, approx never touches verifier, fused == two-pass numpy/torch reference)."
	@echo "NOT checked here: CUDA fused kernel vs reference (needs GPU): see 'make bench' -> run_modal.py check, and RESULTS.md for recorded GPU checks."

plots:
	$(PY) scripts/plots/plot_all.py

bench:
	@echo "# These commands cost GPU money (Modal list price, see results/SPEND.md). They are NOT run by make."
	@echo "# Always --detach so a dropped laptop does not kill the run; log each launch in RUNS.md."
	@echo "# One-time (CPU only): populate the weights volume"
	@echo "modal run --detach modal/download.py"
	@echo "# Phase 0/1 (A100-80GB, eager HF/RefNorth): baseline, ceiling, routing, self-spec -> results/{baseline,ceiling,routing,spec}.jsonl"
	@echo "modal run --detach modal/run.py::baseline"
	@echo "modal run --detach modal/run.py::ceiling"
	@echo "modal run --detach modal/run.py::routing"
	@echo "modal run --detach modal/run.py::spec"
	@echo "# H100 (cap 4 GPU-h / ~16 USD): graphed block verify + overlap traces, graphed self-spec, cohere-megakernel plain decode"
	@echo "modal run --detach modal/h100.py::blocks"
	@echo "modal run --detach modal/h100.py::spec"
	@echo "modal run --detach modal/mk.py::main --which all"
	@echo "# Fused vs two-pass MoE microbench: A100 synthetic sweep, then H100 replay of real routing (build replay input first: python harness/make_replay.py)"
	@echo "modal run --detach modal/run_modal.py::main --full true --seeds 1 --tag fullsweep"
	@echo "modal run --detach modal/run_modal.py::replay --tag real"
	@echo "# Self-improvement loop (live North proposer, GPU evaluator; --max-usd caps spend)"
	@echo "modal run --detach modal/selfimprove_eval.py::remote --track kernel --proposer north --mode structured --rounds 3 --max-usd 5 --confirm-cost --store selfimprove_kernel_structured2"
	@echo "modal run modal/selfimprove_eval.py::pull --store selfimprove_kernel_structured2   # CPU: copy Volume -> results/"
	@echo "# Draft-approximation sweep (A100-80GB)"
	@echo "modal run --detach src/approx/modal_sweep.py::main --smoke true"

all: test check plots
