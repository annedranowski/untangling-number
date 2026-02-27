#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import smtplib
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional

import gymnasium as gym
import matplotlib
import numpy as np
from gymnasium import spaces
from spherogram import Link
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# -----------------------------
# Source-of-truth env helpers
# -----------------------------

SEED = 42

_RE_DT_PREFIX = re.compile(r"^\s*DT\s*:\s*\[", re.I)
_RE_PDLIST = re.compile(r"^\s*\[\s*(\[\s*\d+(?:\s*,\s*\d+){3}\s*\]\s*,?\s*)+\]\s*$")
_RE_XPD = re.compile(r"[Xx]\s*\[")


def parse_link_strict(s: str) -> Link:
    t = s.strip()
    if _RE_DT_PREFIX.match(t):
        return Link(t)
    if _RE_PDLIST.match(t):
        try:
            pd_obj = json.loads(t)
        except json.JSONDecodeError:
            import ast

            pd_obj = ast.literal_eval(t)
        return Link(pd_obj)
    if _RE_XPD.search(t):
        try:
            return Link(t)
        except Exception:
            items = re.findall(r"[Xx]\s*\[([^\]]+)\]", t)
            if not items:
                raise
            blocks = []
            for it in items:
                nums = [int(x.strip()) for x in it.split(",")]
                if len(nums) != 4:
                    raise ValueError("PD block must have 4 integers")
                blocks.append(nums)
            return Link(str(blocks))
    if (t.startswith("{") or t.startswith("[")) and not _RE_PDLIST.match(t):
        try:
            obj = json.loads(t)
            for key in ("pd", "PD", "pd_code", "PD_code", "dt", "DT"):
                if key in obj:
                    return parse_link_strict(obj[key])
        except Exception:
            pass
    raise ValueError("Not a PD/DT code")


def crossings(link: Link) -> int:
    return len(link.crossings)


def is_trivial_zero(link: Link) -> bool:
    return crossings(link) == 0


def riii_shuffle_only_link(link: Link, k: int, tries_per_move: int = 20):
    from spherogram.links import simplify as _simp

    list_fn = getattr(_simp, "possible_type_III_moves", None)
    apply_fn = getattr(_simp, "reidemeister_III", None)
    if list_fn is None or apply_fn is None:
        return link, 0

    done = 0
    for _ in range(k):
        moves = list_fn(link)
        if not moves:
            break
        tries = min(tries_per_move, len(moves))
        c0 = crossings(link)
        success = False
        for tri in random.sample(moves, tries):
            apply_fn(link, tri)
            if crossings(link) == c0:
                success = True
                break
        if not success:
            break
        done += 1
    return link, done


maxstepsdone = 500


@dataclass
class EnvCfg:
    max_steps: int = maxstepsdone
    step_penalty: float = 0.05
    reward_finish: float = 10.0
    allow_backtrack: bool = True
    cap_max: int = 8
    mode_rewards: tuple[float, float, float, float] = (3.0, 2.0, 1.0, 0.0)


class SphKnotEnv(gym.Env):
    def __init__(self, pd_lines: list[str], cfg: EnvCfg):
        super().__init__()
        self.cfg = cfg
        self.pd_lines = pd_lines
        self.rng = random.Random(SEED)

        self.num_actions = 4 if self.cfg.allow_backtrack else 3
        self.action_space = spaces.MultiDiscrete(
            np.array([self.num_actions, self.cfg.cap_max + 1], dtype=np.int64)
        )
        self.obs_dim = 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.L: Optional[Link] = None
        self._steps = 0
        self._last_drop = 0
        self._after_backtrack = False
        self._blocked = [False, False, False, False]

    def _reset_blocks(self):
        self._blocked = [False, False, False, False]

    def _map_blocked_mode(self, mode: int) -> int:
        m = mode % self.num_actions
        for _ in range(self.num_actions):
            if not self._blocked[m]:
                return m
            m = (m + 1) % self.num_actions
        return min(3, self.num_actions - 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self._last_drop = 0
        self._after_backtrack = False
        self._reset_blocks()
        for _ in range(10):
            s = self.rng.choice(self.pd_lines)
            try:
                self.L = parse_link_strict(s)
                break
            except Exception:
                self.L = None
        if self.L is None:
            self.L = parse_link_strict(self.pd_lines[0])
        return self._obs(), {"crossings": crossings(self.L)}

    def step(self, action):
        self._steps += 1
        if isinstance(action, (list, tuple, np.ndarray)):
            mode_req, cap = int(action[0]), int(action[1])
        else:
            mode_req, cap = int(action), 0
        cap = max(0, min(cap, self.cfg.cap_max))

        mode = self._map_blocked_mode(mode_req)
        c_before = crossings(self.L)

        if mode == 0:
            self.L.simplify(mode="basic")
        elif mode == 1:
            steps = cap if cap > 0 else 1
            self.L.simplify(mode="level", type_III_limit=steps)
        elif mode == 2:
            steps = cap if cap > 0 else 1
            self.L.simplify(mode="pickup", type_III_limit=steps)
        elif mode == 3 and self.num_actions == 4:
            steps = cap if cap > 0 else 1
            self.L.backtrack(steps=steps, prob_type_1=0.35, prob_type_2=0.65)
            self.L, _ = riii_shuffle_only_link(self.L, min(steps, 2))

        c_after = crossings(self.L)
        delta = c_before - c_after
        self._last_drop = max(delta, 0)

        reward = self.cfg.mode_rewards[mode] - self.cfg.step_penalty

        done = False
        if is_trivial_zero(self.L):
            reward += self.cfg.reward_finish
            done = True
        if self._steps >= self.cfg.max_steps:
            done = True

        if delta > 0:
            self._reset_blocks()
        else:
            if mode == 3:
                self._reset_blocks()
            else:
                self._blocked[mode] = True

        self._after_backtrack = mode == 3 and self.num_actions == 4

        info = {
            "crossings": c_after,
            "delta": delta,
            "mode_requested": mode_req,
            "mode_effective": mode,
            "cap": cap,
            "blocked": tuple(self._blocked),
        }
        return self._obs(), reward, done, False, info


def _obs_patch(self):
    c = crossings(self.L)
    try:
        comps = len(self.L.link_components)
    except Exception:
        comps = 1
    tmp = Link(self.L.PD_code())
    reduced = tmp.simplify(mode="basic")
    can_reduce = 1.0 if (reduced and crossings(tmp) < c) else 0.0
    recent = 1.0 if getattr(self, "_last_drop", 0) > 0 else 0.0
    return np.array([c, comps, self._steps, can_reduce, recent, 1.0], dtype=np.float32)


SphKnotEnv._obs = getattr(SphKnotEnv, "_obs", _obs_patch)

# -----------------------------
# Variant generation + eval
# -----------------------------


def parse_pd_line_to_list(line: str):
    t = line.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple)):
            out = []
            for quad in obj:
                if len(quad) != 4:
                    raise ValueError(f"PD block must have 4 integers, got {quad!r}")
                out.append([int(x) for x in quad])
            return out
    except Exception:
        pass

    items = re.findall(r"[Xx]\s*\[([^\]]+)\]", t)
    if not items:
        raise ValueError(f"No X[...] blocks found in line: {line!r}")

    out = []
    for it in items:
        nums = [int(x.strip()) for x in it.split(",")]
        if len(nums) != 4:
            raise ValueError(f"PD block must have 4 integers, got {nums!r}")
        out.append(nums)
    return out


def _normalize_pd_cell(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        t = value.strip()
        return t or None
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value).strip() or None


def read_pd_lines_from_file(
    path: Path,
    max_lines: int | None = None,
    *,
    pd_column: str = "pd_json",
):
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "Reading Parquet PD pools requires 'pyarrow'. Add it to the env and retry."
            ) from e

        table = pq.read_table(path)
        cols = set(table.column_names)
        if pd_column not in cols:
            fallback = [name for name in ("pd_json", "pd", "pd_code") if name in cols]
            if not fallback:
                raise ValueError(
                    f"Parquet file {path} is missing PD column '{pd_column}'. "
                    f"Available columns: {sorted(cols)}"
                )
            pd_column = fallback[0]

        lines = []
        for val in table.column(pd_column).to_pylist():
            t = _normalize_pd_cell(val)
            if t:
                lines.append(t)
            if max_lines is not None and len(lines) >= max_lines:
                break
        return lines

    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[:max_lines]


def flip_crossing_quad(quad):
    a, b, c, d = quad
    return [b, c, d, a]


def apply_flips(pd_list, flipped_indices):
    idxs_set = set(flipped_indices)
    out = []
    for i, quad in enumerate(pd_list):
        if i in idxs_set:
            out.append(flip_crossing_quad(quad))
        else:
            out.append(list(quad))
    return out


def pd_list_to_str(pd_list):
    return json.dumps(pd_list)


def make_single_env(pd_list, cfg: EnvCfg):
    pd_lines_single = [pd_list_to_str(pd_list)]

    def _make():
        return SphKnotEnv(pd_lines_single, cfg)

    return DummyVecEnv([_make])


def run_unknotter_on_pd(
    pd_list,
    model,
    cfg: EnvCfg,
    *,
    episodes: int = 3,
    deterministic: bool = True,
) -> tuple[bool, int, int]:
    vec_env = make_single_env(pd_list, cfg)
    success = False
    min_crossings = len(pd_list)
    steps_taken_total = 0

    for _ep in range(episodes):
        obs = vec_env.reset()
        for _step in range(cfg.max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            cr = info.get("crossings", None)
            if cr is not None:
                min_crossings = min(min_crossings, int(cr))
            steps_taken_total += 1

            if cr == 0:
                success = True
                min_crossings = 0
                break
            if dones[0]:
                break
        if success:
            break

    vec_env.close()
    return success, min_crossings, steps_taken_total


# -----------------------------
# Worker globals and functions
# -----------------------------

MODEL = None
WORKER_CFG = None
WORKER_EPISODES = 1
WORKER_DETERMINISTIC = True


def init_worker(
    model_path: str,
    cfg_payload: dict,
    episodes: int,
    deterministic: bool,
    seed: int,
):
    global MODEL, WORKER_CFG, WORKER_EPISODES, WORKER_DETERMINISTIC, SEED
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    WORKER_CFG = EnvCfg(**cfg_payload)
    WORKER_EPISODES = episodes
    WORKER_DETERMINISTIC = deterministic
    MODEL = PPO.load(model_path, device="cpu")


def eval_chunk(chunk):
    rows = []
    for item in chunk:
        source_idx = item["source_index"]
        variant_id = item["variant_id"]
        flipped_indices = tuple(item["flipped_indices"])
        original_pd = item["original_pd"]

        flipped_pd = apply_flips(original_pd, flipped_indices)
        success, min_cr, steps_taken = run_unknotter_on_pd(
            flipped_pd,
            MODEL,
            WORKER_CFG,
            episodes=WORKER_EPISODES,
            deterministic=WORKER_DETERMINISTIC,
        )
        rows.append(
            {
                "variant_id": variant_id,
                "source_index": source_idx,
                "original_crossings": len(original_pd),
                "flipped_indices": list(flipped_indices),
                "episodes": WORKER_EPISODES,
                "deterministic": WORKER_DETERMINISTIC,
                "rl_unknot_success": bool(success),
                "min_crossings_found": int(min_cr),
                "steps_taken_total": int(steps_taken),
            }
        )
    return rows


# -----------------------------
# IO / reporting
# -----------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def iter_variants(pd_lists, flips: int):
    for source_index, pd_list in enumerate(pd_lists):
        n_crossings = len(pd_list)
        if n_crossings < flips:
            continue
        for idxs in itertools.combinations(range(n_crossings), flips):
            idxs_tuple = tuple(int(i) for i in idxs)
            variant_id = f"{source_index}:{','.join(map(str, idxs_tuple))}"
            yield {
                "variant_id": variant_id,
                "source_index": source_index,
                "original_pd": pd_list,
                "flipped_indices": idxs_tuple,
            }


def write_jsonl_row(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def append_csv_rows(path: Path, rows: list[dict], write_header: bool):
    if not rows:
        return
    fieldnames = [
        "variant_id",
        "source_index",
        "original_crossings",
        "flipped_indices",
        "episodes",
        "deterministic",
        "rl_unknot_success",
        "min_crossings_found",
        "steps_taken_total",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            out = dict(row)
            out["flipped_indices"] = json.dumps(out["flipped_indices"])
            writer.writerow(out)


def load_processed_ids(results_jsonl: Path) -> set[str]:
    if not results_jsonl.exists():
        return set()
    out: set[str] = set()
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vid = obj.get("variant_id")
                if isinstance(vid, str):
                    out.add(vid)
            except Exception:
                continue
    return out


def load_all_results(results_jsonl: Path) -> list[dict]:
    out = []
    if not results_jsonl.exists():
        return out
    with results_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_summary_and_hist(run_dir: Path, all_results: list[dict]):
    summary_csv = run_dir / "summary.csv"
    hist_png = run_dir / "hist_min_crossings.png"

    total = len(all_results)
    if total == 0:
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_variants", 0])
        return

    min_vals = [int(r["min_crossings_found"]) for r in all_results]
    success_vals = [1 if r.get("rl_unknot_success") else 0 for r in all_results]

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_variants", total])
        writer.writerow(["success_count", int(sum(success_vals))])
        writer.writerow(["success_rate", float(np.mean(success_vals))])
        writer.writerow(["mean_min_crossings", float(np.mean(min_vals))])
        writer.writerow(["median_min_crossings", float(np.median(min_vals))])
        writer.writerow(["min_min_crossings", int(np.min(min_vals))])
        writer.writerow(["max_min_crossings", int(np.max(min_vals))])
        writer.writerow(["count_min_crossings_0", int(sum(1 for x in min_vals if x == 0))])
        writer.writerow(["count_min_crossings_le_3", int(sum(1 for x in min_vals if x <= 3))])

    plt.figure(figsize=(12, 6))
    bins = range(0, max(min_vals) + 2)
    plt.hist(min_vals, bins=bins, edgecolor="black")
    plt.title("Distribution of Minimum Crossings Found by RL Model")
    plt.xlabel("Minimum Crossings Found")
    plt.ylabel("Frequency")
    plt.xticks(range(0, max(min_vals) + 1))
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(hist_png, dpi=150)
    plt.close()


def maybe_send_email(
    *,
    run_dir: Path,
    summary_csv: Path,
    hist_png: Path,
    progress: dict,
    email_to: str | None,
):
    if not email_to:
        return

    smtp_host = os.getenv("SWEEP_SMTP_HOST")
    smtp_port = int(os.getenv("SWEEP_SMTP_PORT", "587"))
    smtp_user = os.getenv("SWEEP_SMTP_USER")
    smtp_pass = os.getenv("SWEEP_SMTP_PASS")
    smtp_from = os.getenv("SWEEP_EMAIL_FROM", smtp_user or "")

    if not (smtp_host and smtp_user and smtp_pass and smtp_from):
        print("[email] Skipped: SMTP env vars not fully configured.")
        return

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = email_to
    msg["Subject"] = f"Knot sweep complete: {run_dir.name}"
    msg.set_content(
        "Sweep complete.\n\n"
        f"Run dir: {run_dir}\n"
        f"Processed: {progress.get('processed_total')}\n"
        f"Success rate: {progress.get('success_rate')}\n"
    )

    for fp in [summary_csv, hist_png]:
        if not fp.exists():
            continue
        data = fp.read_bytes()
        maintype, subtype = ("text", "csv") if fp.suffix == ".csv" else ("image", "png")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=fp.name)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"[email] Sent completion email to {email_to}")
    except Exception as e:
        # Email is best-effort; do not fail the whole run at completion.
        print(f"[email] Failed to send completion email: {type(e).__name__}: {e}")


def chunked(iterable, chunk_size: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= chunk_size:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    parser = argparse.ArgumentParser(description="Parallel variant sweep with checkpoint/resume.")
    parser.add_argument("--base", default="crossing-reduction")
    parser.add_argument("--pd-path", default=None)
    parser.add_argument("--pd-column", default="pd_json")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--flips", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--allow-backtrack", dest="allow_backtrack", action="store_true")
    parser.add_argument("--no-backtrack", dest="allow_backtrack", action="store_false")
    parser.set_defaults(allow_backtrack=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    parser.add_argument("--tqdm", dest="use_tqdm", action="store_true")
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.set_defaults(use_tqdm=True)
    parser.add_argument("--max-pds", type=int, default=None)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume-dir", default=None)
    parser.add_argument("--email-to", default=None)
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    base = Path(args.base).resolve()
    pd_path = Path(args.pd_path).resolve() if args.pd_path else (base / "generated_T_pd_backtrack.txt")
    model_path = Path(args.model_path).resolve() if args.model_path else (base / "best_model.zip")
    output_root = Path(args.output_root).resolve() if args.output_root else (base / "runs")

    if args.resume_dir:
        run_dir = Path(args.resume_dir).resolve()
    else:
        run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = run_dir / "results.jsonl"
    results_csv = run_dir / "results.csv"
    progress_json = run_dir / "progress.json"
    metadata_json = run_dir / "metadata.json"

    cfg = EnvCfg(max_steps=args.max_steps, allow_backtrack=args.allow_backtrack)
    metadata = {
        "started_at_utc": utc_now_iso(),
        "base": str(base),
        "pd_path": str(pd_path),
        "pd_column": args.pd_column,
        "model_path": str(model_path),
        "git_hash": safe_git_hash(),
        "seed": args.seed,
        "flips": args.flips,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "use_tqdm": args.use_tqdm,
        "max_pds": args.max_pds,
        "max_variants": args.max_variants,
        "cfg": asdict(cfg),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    raw_pd_lines = read_pd_lines_from_file(pd_path, max_lines=args.max_pds, pd_column=args.pd_column)
    pd_lists = [parse_pd_line_to_list(s) for s in raw_pd_lines]
    total_possible = sum(n_choose_k(len(p), args.flips) for p in pd_lists if len(p) >= args.flips)
    if args.max_variants is not None:
        total_possible = min(total_possible, args.max_variants)

    processed_ids = load_processed_ids(results_jsonl)
    print(f"[resume] Found {len(processed_ids)} processed variant_ids in {results_jsonl}")
    print(f"[plan] Total variants target: {total_possible}")

    variant_iter = iter_variants(pd_lists, args.flips)
    if args.max_variants is not None:
        variant_iter = itertools.islice(variant_iter, args.max_variants)

    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    if total_possible == 0:
        print("[done] No variants generated.")
        return

    chunks_generated = 0
    chunks_to_run = 0
    for chunk in chunked(variant_iter, args.chunk_size):
        chunks_generated += 1
        if any(item["variant_id"] not in processed_ids for item in chunk):
            chunks_to_run += 1

    # Rebuild iterator for actual execution
    variant_iter = iter_variants(pd_lists, args.flips)
    if args.max_variants is not None:
        variant_iter = itertools.islice(variant_iter, args.max_variants)

    if chunks_generated == 0:
        print("[done] No variants generated.")
        return
    if chunks_to_run == 0:
        print("[done] Nothing to run; all variants already processed.")
    else:

        started = time.time()
        processed_total = len(processed_ids)
        success_total = 0
        log_next = processed_total + args.log_every
        last_heartbeat = 0.0
        use_tqdm = bool(args.use_tqdm)

        cfg_payload = asdict(cfg)

        def write_progress(force: bool = False):
            nonlocal last_heartbeat
            now = time.time()
            if not force and (now - last_heartbeat) < args.heartbeat_seconds:
                return
            elapsed = max(now - started, 1e-9)
            rate = (processed_total - len(processed_ids)) / elapsed
            remaining = max(total_possible - processed_total, 0)
            eta_seconds = int(remaining / rate) if rate > 1e-9 else None
            success_rate = (success_total / processed_total) if processed_total else 0.0
            payload = {
                "timestamp_utc": utc_now_iso(),
                "processed_total": processed_total,
                "total_target": total_possible,
                "remaining": remaining,
                "success_total": success_total,
                "success_rate": success_rate,
                "rate_variants_per_sec": rate,
                "eta_seconds": eta_seconds,
            }
            progress_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            last_heartbeat = now

        csv_has_header = results_csv.exists() and results_csv.stat().st_size > 0

        pbar = (
            tqdm(
                total=total_possible,
                initial=min(processed_total, total_possible),
                desc="variant sweep",
                unit="variant",
                dynamic_ncols=True,
                mininterval=1.0,
            )
            if use_tqdm
            else None
        )
        try:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker,
                initargs=(str(model_path), cfg_payload, args.episodes, args.deterministic, args.seed),
            ) as pool:
                max_in_flight = max(1, args.workers * 4)
                in_flight = set()

                def submit_more():
                    while len(in_flight) < max_in_flight:
                        try:
                            chunk = next(chunks_iter)
                        except StopIteration:
                            break
                        filtered = [item for item in chunk if item["variant_id"] not in processed_ids]
                        if not filtered:
                            continue
                        in_flight.add(pool.submit(eval_chunk, filtered))

                chunks_iter = iter(chunked(variant_iter, args.chunk_size))
                submit_more()

                while in_flight:
                    done, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in done:
                        in_flight.remove(fut)
                        rows = fut.result()
                        for row in rows:
                            write_jsonl_row(results_jsonl, row)
                        append_csv_rows(results_csv, rows, write_header=not csv_has_header)
                        csv_has_header = True

                        processed_total += len(rows)
                        success_total += sum(1 for r in rows if r.get("rl_unknot_success"))
                        if pbar is not None:
                            pbar.update(len(rows))

                        if processed_total >= log_next:
                            elapsed = max(time.time() - started, 1e-9)
                            rate = (processed_total - len(processed_ids)) / elapsed
                            success_rate = success_total / max(processed_total, 1)
                            print(
                                f"[progress] processed={processed_total}/{total_possible} "
                                f"rate={rate:.2f}/s success_rate={success_rate:.4f}"
                            )
                            if pbar is not None:
                                pbar.set_postfix(
                                    {
                                        "rate/s": f"{rate:.2f}",
                                        "success": f"{success_rate:.4f}",
                                    },
                                    refresh=False,
                                )
                            log_next += args.log_every

                        write_progress(force=False)
                        submit_more()
        finally:
            if pbar is not None:
                pbar.close()

        write_progress(force=True)

    all_results = load_all_results(results_jsonl)
    write_summary_and_hist(run_dir, all_results)

    summary_csv = run_dir / "summary.csv"
    hist_png = run_dir / "hist_min_crossings.png"
    final_progress = json.loads(progress_json.read_text(encoding="utf-8")) if progress_json.exists() else {}
    maybe_send_email(
        run_dir=run_dir,
        summary_csv=summary_csv,
        hist_png=hist_png,
        progress=final_progress,
        email_to=args.email_to,
    )

    print(f"[done] run_dir={run_dir}")
    print(f"[done] results_jsonl={results_jsonl}")
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] hist_png={hist_png}")


if __name__ == "__main__":
    main()
