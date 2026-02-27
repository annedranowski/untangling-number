# Untangling Number

Reproducible workflows for RL-based knot crossing reduction and variant sweep analysis.

## What This Repo Contains

- `crossing-reduction/run_variant_sweep.py`: parallel sweep over flipped crossing variants.
- `crossing-reduction/generate_t_backtrack_pool.py`: generate deduplicated PD pools (TXT/Parquet).
- `crossing-reduction/summarize_success.py`: descriptive success summaries (base PD + flips + crossings).
- `crossing-reduction/explore_variant_sweep_results.ipynb`: analysis notebook with visualization cells.

## Prerequisites

- Python `3.13.x` (project is pinned to `>=3.13,<3.14`)
- `uv` package manager: https://docs.astral.sh/uv/

## Quickstart

```bash
make setup
```

This installs runtime + dev dependencies and uses `uv.lock` for reproducible installs.

## Common Team Commands

### 1) Smoke test (fast)

```bash
make smoke
```

Writes outputs under `crossing-reduction/runs_smoke/<timestamp>/`.

### 2) Full sweep

```bash
make sweep
```

By default this uses:
- PD pool: `crossing-reduction/generated_T_pd_backtrack_1M.parquet`
- model: `crossing-reduction/best_model.zip`
- `--flips 5`

Outputs are written to `crossing-reduction/runs/<timestamp>/`.

### 3) Build descriptive success summary for a run

```bash
make summarize RUN_DIR=crossing-reduction/runs/<timestamp>
```

Creates these files in `<RUN_DIR>/analysis/`:
- `success_summary_descriptive_by_base.csv`
- `success_top_flips_by_base.csv`
- `success_flip_index_enrichment_by_base.csv`
- `success_summary_descriptive.md`

### 4) Open notebook

```bash
make notebook
```

Run section `10) Success Summary Visualizations` in:
- `crossing-reduction/explore_variant_sweep_results.ipynb`

## Direct CLI Examples

### Sweep with explicit limits

```bash
uv run python crossing-reduction/run_variant_sweep.py \
  --base crossing-reduction \
  --pd-path crossing-reduction/generated_T_pd_backtrack_1M.parquet \
  --flips 5 \
  --max-pds 10 \
  --max-variants 10000
```

### Generate a new PD pool

```bash
uv run python crossing-reduction/generate_t_backtrack_pool.py \
  --out-parquet crossing-reduction/generated_T_pd_backtrack_1M.parquet \
  --out-txt crossing-reduction/generated_T_pd_backtrack_1M.txt \
  --target-flips 5 \
  --target-variants 1000000
```

### Summarize a run

```bash
uv run python crossing-reduction/summarize_success.py \
  --run-dir crossing-reduction/runs/<timestamp>
```

## Notes

- Large/generated artifacts are intentionally git-ignored:
  - `crossing-reduction/3-16.txt`
  - `crossing-reduction/runs/`
  - `crossing-reduction/runs_smoke/`
  - `crossing-reduction/tb_knots/`
  - `crossing-reduction/runs/*.log`
- Keep heavyweight data/artifacts in external storage (Drive/S3) and share links in PRs/issues when needed.
- `summarize_success.py` can use base PDs from either:
  - existing `<RUN_DIR>/analysis/success_zero_variants.csv`, or
  - `metadata.json` PD pool path.
