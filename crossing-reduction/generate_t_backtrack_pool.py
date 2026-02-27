#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from spherogram import Link

# Same base pair used in rluntanglenumber.ipynb to construct T = K1 # K2.
K1_DEFAULT = [[4, 2, 5, 1], [8, 6, 1, 5], [6, 3, 7, 4], [2, 7, 3, 8]]
K2_DEFAULT = [
    [1, 13, 2, 12],
    [3, 11, 4, 10],
    [5, 17, 6, 16],
    [7, 15, 8, 14],
    [9, 1, 10, 18],
    [11, 3, 12, 2],
    [13, 5, 14, 4],
    [15, 7, 16, 6],
    [17, 9, 18, 8],
]


def connected_sum_pd(pd1, pd2, simplify: bool = True) -> list[list[int]]:
    l1, l2 = Link(pd1), Link(pd2)
    out = l1.connected_sum(l2)
    l = out if out is not None else l1
    if simplify and hasattr(l, "simplify"):
        try:
            l.simplify()
        except TypeError:
            pass
    return normalize_pd_code(l.PD_code())


def normalize_pd_code(pd_obj) -> list[list[int]]:
    if isinstance(pd_obj, list):
        out = []
        for quad in pd_obj:
            if len(quad) != 4:
                raise ValueError(f"Expected 4 entries per crossing, got {quad!r}")
            out.append([int(getattr(x, "label", x)) for x in quad])
        return out
    out = []
    for vtx in pd_obj:
        quad = [int(getattr(edge, "label", edge)) for edge in vtx]
        if len(quad) != 4:
            raise ValueError(f"Expected 4 entries per crossing, got {quad!r}")
        out.append(quad)
    return out


def riii_shuffle_only_link(link: Link, k: int, tries_per_move: int = 20):
    from spherogram.links import simplify as simp

    list_fn = getattr(simp, "possible_type_III_moves", None)
    apply_fn = getattr(simp, "reidemeister_III", None)
    if list_fn is None or apply_fn is None:
        return link, 0

    done = 0
    for _ in range(k):
        moves = list_fn(link)
        if not moves:
            break
        tries = min(tries_per_move, len(moves))
        c0 = len(link.crossings)
        success = False
        for tri in random.sample(moves, tries):
            apply_fn(link, tri)
            if len(link.crossings) == c0:
                success = True
                break
        if not success:
            break
        done += 1
    return link, done


def write_parquet(path: Path, rows: list[dict], compression: str):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "Writing Parquet requires 'pyarrow'. Install project deps and retry."
        ) from e

    table = pa.table(
        {
            "pd_json": [r["pd_json"] for r in rows],
            "crossings": [r["crossings"] for r in rows],
            "backtrack_steps": [r["backtrack_steps"] for r in rows],
            "riii_done": [r["riii_done"] for r in rows],
            "attempt": [r["attempt"] for r in rows],
        }
    )
    pq.write_table(table, path, compression=compression)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a large deduplicated backtrack pool for T and save as Parquet."
    )
    p.add_argument(
        "--out-parquet",
        default="crossing-reduction/generated_T_pd_backtrack_1M.parquet",
        help="Output Parquet path.",
    )
    p.add_argument(
        "--out-txt",
        default=None,
        help="Optional newline-delimited PD JSON output path.",
    )
    p.add_argument(
        "--target-flips",
        type=int,
        default=5,
        help="Flip count used to estimate combinatorial sweep size.",
    )
    p.add_argument(
        "--target-variants",
        type=int,
        default=1_000_000,
        help="Stop once estimated total variants for --target-flips reaches this count.",
    )
    p.add_argument("--max-attempts", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backtrack-min", type=int, default=1)
    p.add_argument("--backtrack-max", type=int, default=8)
    p.add_argument("--riii-max", type=int, default=48)
    p.add_argument("--min-crossings-keep", type=int, default=15)
    p.add_argument("--max-crossings-keep", type=int, default=28)
    p.add_argument("--status-every", type=int, default=100)
    p.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli"])
    return p.parse_args()


def main():
    args = parse_args()
    if args.target_flips < 0:
        raise ValueError("--target-flips must be non-negative")
    if args.backtrack_min <= 0 or args.backtrack_max < args.backtrack_min:
        raise ValueError("invalid backtrack bounds")
    if args.riii_max <= 0:
        raise ValueError("--riii-max must be > 0")

    random.seed(args.seed)

    out_parquet = Path(args.out_parquet).resolve()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_txt = Path(args.out_txt).resolve() if args.out_txt else None
    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)

    t_pd = connected_sum_pd(K1_DEFAULT, K2_DEFAULT, simplify=True)
    t_crossings = len(t_pd)
    print(f"[init] base crossings(T)={t_crossings}")
    print(
        f"[init] target: flips={args.target_flips} variants>={args.target_variants:,} "
        f"window=[{args.min_crossings_keep},{args.max_crossings_keep}]"
    )

    seen: set[str] = set()
    rows: list[dict] = []
    comb_total = 0

    attempts = 0
    while attempts < args.max_attempts and comb_total < args.target_variants:
        attempts += 1

        l = Link(t_pd)
        steps = random.randint(args.backtrack_min, args.backtrack_max)
        l.backtrack(steps=steps, prob_type_1=0.35, prob_type_2=0.65)
        riii_k = random.randint(1, args.riii_max)
        l, riii_done = riii_shuffle_only_link(l, riii_k)

        quads = normalize_pd_code(l.PD_code())
        c = len(quads)
        if c < args.min_crossings_keep or c > args.max_crossings_keep:
            continue

        pd_json = json.dumps(quads, separators=(",", ":"))
        if pd_json in seen:
            continue
        seen.add(pd_json)

        increment = math.comb(c, args.target_flips) if c >= args.target_flips else 0
        comb_total += increment
        rows.append(
            {
                "pd_json": pd_json,
                "crossings": int(c),
                "backtrack_steps": int(steps),
                "riii_done": int(riii_done),
                "attempt": int(attempts),
            }
        )

        if len(rows) == 1 or len(rows) % args.status_every == 0:
            print(
                f"[progress] accepted={len(rows)} attempts={attempts} "
                f"comb_total={comb_total:,} last_crossings={c} "
                f"last_comb={increment:,}"
            )

    coverage = comb_total / max(args.target_variants, 1)
    print(
        f"[done] accepted={len(rows)} attempts={attempts} "
        f"comb_total={comb_total:,} ({coverage:.3f}x target)"
    )
    if comb_total < args.target_variants:
        print("[warn] max attempts reached before target combinatorial coverage.")

    write_parquet(out_parquet, rows, compression=args.compression)
    print(f"[done] wrote parquet={out_parquet}")

    if out_txt is not None:
        with out_txt.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(row["pd_json"] + "\n")
        print(f"[done] wrote txt={out_txt}")


if __name__ == "__main__":
    main()
