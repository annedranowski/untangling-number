#!/usr/bin/env python3
"""Generate descriptive success summaries for a variant sweep run."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_flips(raw: str) -> tuple[int, ...]:
    val = ast.literal_eval(raw)
    if isinstance(val, list):
        return tuple(int(x) for x in val)
    raise ValueError(f"Expected list for flipped_indices, got: {raw}")


def parse_pd_text_to_json(line: str) -> str:
    t = line.strip()
    if not t:
        raise ValueError("Empty PD line")
    if t.startswith("[["):
        quads = json.loads(t)
        return json.dumps(quads, separators=(",", ":"))
    if t.startswith("PD["):
        nums: list[int] = []
        cur = ""
        for ch in t:
            if ch.isdigit():
                cur += ch
            elif cur:
                nums.append(int(cur))
                cur = ""
        if cur:
            nums.append(int(cur))
        if len(nums) % 4 != 0:
            raise ValueError(f"Could not parse PD line: {line[:80]}...")
        quads = [nums[i : i + 4] for i in range(0, len(nums), 4)]
        return json.dumps(quads, separators=(",", ":"))
    raise ValueError(f"Unsupported PD format: {t[:32]}")


def read_pd_json_lines(path: Path, pd_column: str) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading parquet requires pyarrow") from exc
        table = pq.read_table(path)
        cols = set(table.column_names)
        col = pd_column if pd_column in cols else ("pd_json" if "pd_json" in cols else None)
        if col is None:
            raise KeyError(f"No PD column found in parquet. columns={sorted(cols)}")
        return [str(v.as_py()).strip() for v in table[col]]

    pd_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            pd_lines.append(parse_pd_text_to_json(t))
    return pd_lines


def load_base_pd_by_source(run_dir: Path, pd_path: Path | None, pd_column: str) -> dict[int, str]:
    success_zero = run_dir / "analysis" / "success_zero_variants.csv"
    if success_zero.exists():
        mapping: dict[int, str] = {}
        with success_zero.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames and "base_pd_json" in r.fieldnames:
                for row in r:
                    s = int(row["source_index"])
                    if s not in mapping and row.get("base_pd_json"):
                        mapping[s] = row["base_pd_json"]
                if mapping:
                    return mapping

    if pd_path is None:
        return {}

    pd_lines = read_pd_json_lines(pd_path, pd_column)
    return {idx: line for idx, line in enumerate(pd_lines)}


def mean_or_blank(vals: list[int]) -> float | str:
    return round(sum(vals) / len(vals), 3) if vals else ""


def median_or_blank(vals: list[int]) -> float | str:
    return round(statistics.median(vals), 3) if vals else ""


def min_or_blank(vals: list[int]) -> int | str:
    return min(vals) if vals else ""


def max_or_blank(vals: list[int]) -> int | str:
    return max(vals) if vals else ""


def top_flips(counter: Counter[tuple[int, ...]], k: int = 5) -> str:
    return " | ".join(f"{list(f)} ({c})" for f, c in counter.most_common(k))


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize success patterns for a variant sweep run.")
    p.add_argument("--run-dir", required=True, help="Run directory containing results.csv and metadata.json")
    p.add_argument("--pd-path", default=None, help="Optional PD pool override (otherwise metadata pd_path)")
    p.add_argument("--pd-column", default=None, help="Parquet PD column name (default from metadata or pd_json)")
    p.add_argument("--analysis-dir", default=None, help="Output analysis directory (default <run-dir>/analysis)")
    p.add_argument("--min-enrichment-count", type=int, default=100, help="Keep indices with at least this many observations in enrichment table")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    results_path = run_dir / "results.csv"
    metadata_path = run_dir / "metadata.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    pd_path = Path(args.pd_path).resolve() if args.pd_path else (
        Path(metadata["pd_path"]).resolve() if metadata.get("pd_path") else None
    )
    pd_column = args.pd_column or metadata.get("pd_column", "pd_json")

    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else (run_dir / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    out_summary = analysis_dir / "success_summary_descriptive_by_base.csv"
    out_top_flips = analysis_dir / "success_top_flips_by_base.csv"
    out_enrich = analysis_dir / "success_flip_index_enrichment_by_base.csv"
    out_md = analysis_dir / "success_summary_descriptive.md"

    base_pd_by_source = load_base_pd_by_source(run_dir, pd_path, pd_column)

    agg: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "success": 0,
            "failure": 0,
            "orig_crossings": [],
            "succ_orig_crossings": [],
            "fail_orig_crossings": [],
            "succ_steps": [],
            "fail_steps": [],
            "fail_min_crossings": [],
            "succ_flip_counter": Counter(),
            "fail_flip_counter": Counter(),
            "succ_examples": [],
            "idx_total": defaultdict(int),
            "idx_succ": defaultdict(int),
            "base_crossings": None,
        }
    )

    with results_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = int(row["source_index"])
            ok = parse_bool(row["rl_unknot_success"])
            oc = int(float(row["original_crossings"]))
            st = int(float(row["steps_taken_total"]))
            mc = int(float(row["min_crossings_found"]))
            flips = parse_flips(row["flipped_indices"])

            a = agg[s]
            a["total"] += 1
            a["orig_crossings"].append(oc)
            if a["base_crossings"] is None:
                a["base_crossings"] = oc

            for idx in flips:
                a["idx_total"][idx] += 1
                if ok:
                    a["idx_succ"][idx] += 1

            if ok:
                a["success"] += 1
                a["succ_orig_crossings"].append(oc)
                a["succ_steps"].append(st)
                a["succ_flip_counter"][flips] += 1
                if len(a["succ_examples"]) < 3:
                    a["succ_examples"].append((row["variant_id"], row["flipped_indices"]))
            else:
                a["failure"] += 1
                a["fail_orig_crossings"].append(oc)
                a["fail_steps"].append(st)
                a["fail_min_crossings"].append(mc)
                a["fail_flip_counter"][flips] += 1

    summary_rows: list[dict[str, Any]] = []
    for s in sorted(agg):
        a = agg[s]
        total = a["total"]
        success = a["success"]
        summary_rows.append(
            {
                "source_index": s,
                "base_crossings": a["base_crossings"] if a["base_crossings"] is not None else "",
                "base_pd_json": base_pd_by_source.get(s, ""),
                "total_variants": total,
                "success_count": success,
                "failure_count": a["failure"],
                "success_rate": round(success / total, 6) if total else "",
                "failure_rate": round(a["failure"] / total, 6) if total else "",
                "orig_crossings_min": min_or_blank(a["orig_crossings"]),
                "orig_crossings_median": median_or_blank(a["orig_crossings"]),
                "orig_crossings_max": max_or_blank(a["orig_crossings"]),
                "success_orig_crossings_min": min_or_blank(a["succ_orig_crossings"]),
                "success_orig_crossings_median": median_or_blank(a["succ_orig_crossings"]),
                "success_orig_crossings_max": max_or_blank(a["succ_orig_crossings"]),
                "avg_steps_success": mean_or_blank(a["succ_steps"]),
                "avg_steps_failure": mean_or_blank(a["fail_steps"]),
                "failure_min_crossings_best": min_or_blank(a["fail_min_crossings"]),
                "failure_min_crossings_median": median_or_blank(a["fail_min_crossings"]),
                "failure_min_crossings_worst": max_or_blank(a["fail_min_crossings"]),
                "unique_success_flip_patterns": len(a["succ_flip_counter"]),
                "top_success_flips": top_flips(a["succ_flip_counter"]),
                "top_failure_flips": top_flips(a["fail_flip_counter"]),
                "sample_success_variants": " | ".join(f"{vid}:{fl}" for vid, fl in a["succ_examples"]),
            }
        )

    summary_fields = [
        "source_index",
        "base_crossings",
        "base_pd_json",
        "total_variants",
        "success_count",
        "failure_count",
        "success_rate",
        "failure_rate",
        "orig_crossings_min",
        "orig_crossings_median",
        "orig_crossings_max",
        "success_orig_crossings_min",
        "success_orig_crossings_median",
        "success_orig_crossings_max",
        "avg_steps_success",
        "avg_steps_failure",
        "failure_min_crossings_best",
        "failure_min_crossings_median",
        "failure_min_crossings_worst",
        "unique_success_flip_patterns",
        "top_success_flips",
        "top_failure_flips",
        "sample_success_variants",
    ]
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    with out_top_flips.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_index", "result_type", "rank", "flip_indices", "count"])
        w.writeheader()
        for s in sorted(agg):
            for rank, (flips, count) in enumerate(agg[s]["succ_flip_counter"].most_common(10), start=1):
                w.writerow({"source_index": s, "result_type": "success", "rank": rank, "flip_indices": list(flips), "count": count})
            for rank, (flips, count) in enumerate(agg[s]["fail_flip_counter"].most_common(10), start=1):
                w.writerow({"source_index": s, "result_type": "failure", "rank": rank, "flip_indices": list(flips), "count": count})

    with out_enrich.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "source_index",
            "base_crossings",
            "flip_index",
            "index_total_count",
            "index_success_count",
            "index_success_rate",
            "base_success_rate",
            "enrichment_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(agg):
            a = agg[s]
            total = a["total"]
            success = a["success"]
            base_rate = success / total if total else math.nan
            for idx in sorted(a["idx_total"]):
                idx_total = a["idx_total"][idx]
                if idx_total < args.min_enrichment_count:
                    continue
                idx_success = a["idx_succ"][idx]
                idx_rate = idx_success / idx_total
                enrich = (idx_rate / base_rate) if base_rate and not math.isnan(base_rate) else math.nan
                w.writerow(
                    {
                        "source_index": s,
                        "base_crossings": a["base_crossings"] if a["base_crossings"] is not None else "",
                        "flip_index": idx,
                        "index_total_count": idx_total,
                        "index_success_count": idx_success,
                        "index_success_rate": round(idx_rate, 6),
                        "base_success_rate": round(base_rate, 6) if not math.isnan(base_rate) else "",
                        "enrichment_ratio": round(enrich, 6) if not math.isnan(enrich) else "",
                    }
                )

    by_rate = sorted(summary_rows, key=lambda x: x["success_rate"], reverse=True)
    by_count = sorted(summary_rows, key=lambda x: x["success_count"], reverse=True)

    lines: list[str] = [
        "# Success Summary (Descriptive)",
        "",
        "Generated from `results.csv`.",
        "",
        f"- Base PDs: {len(summary_rows)}",
        f"- Total successful zero-crossing variants: {sum(r['success_count'] for r in summary_rows)}",
        f"- Total variants evaluated: {sum(r['total_variants'] for r in summary_rows)}",
        "",
        "## Top 5 Base PDs by Success Rate",
        "",
        "| source_index | base_crossings | success_rate | success_count | total_variants |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in by_rate[:5]:
        lines.append(
            f"| {row['source_index']} | {row['base_crossings']} | {row['success_rate']:.4f} | {row['success_count']} | {row['total_variants']} |"
        )
    lines.extend(
        [
            "",
            "## Top 5 Base PDs by Success Count",
            "",
            "| source_index | base_crossings | success_count | success_rate | total_variants |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_count[:5]:
        lines.append(
            f"| {row['source_index']} | {row['base_crossings']} | {row['success_count']} | {row['success_rate']:.4f} | {row['total_variants']} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `success_summary_descriptive_by_base.csv`: main per-base table (includes `base_pd_json`).",
            "- `success_top_flips_by_base.csv`: top repeated exact flip sets for success/failure.",
            "- `success_flip_index_enrichment_by_base.csv`: per-index enrichment signal.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] wrote {out_summary}")
    print(f"[done] wrote {out_top_flips}")
    print(f"[done] wrote {out_enrich}")
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()
