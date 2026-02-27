.PHONY: help setup smoke sweep summarize notebook

RUN_DIR ?=
BASE := crossing-reduction

help:
	@echo "Targets:"
	@echo "  make setup       - install project + dev deps with uv"
	@echo "  make smoke       - run a small sweep smoke test"
	@echo "  make sweep       - run a full sweep using project defaults"
	@echo "  make summarize RUN_DIR=<path> - build descriptive success summaries"
	@echo "  make notebook    - launch Jupyter Lab"

setup:
	uv sync --extra dev

smoke:
	uv run python crossing-reduction/run_variant_sweep.py \
		--base $(BASE) \
		--pd-path crossing-reduction/generated_T_pd_backtrack_1M.parquet \
		--flips 5 \
		--max-pds 3 \
		--max-variants 500 \
		--workers 2 \
		--chunk-size 16 \
		--output-root crossing-reduction/runs_smoke

sweep:
	uv run python crossing-reduction/run_variant_sweep.py \
		--base $(BASE) \
		--pd-path crossing-reduction/generated_T_pd_backtrack_1M.parquet \
		--flips 5

summarize:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR, e.g. make summarize RUN_DIR=crossing-reduction/runs/20260221_221918" && exit 1)
	uv run python crossing-reduction/summarize_success.py --run-dir $(RUN_DIR)

notebook:
	uv run jupyter lab
