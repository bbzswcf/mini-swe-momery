"""Summarize the last LLM input_tokens before each task completed.

Walks `results/<run>/instance_*/*.traj.json`, finds the last assistant turn that
recorded `usage.input_tokens`, and reports per-run distribution plus a breakdown
by `exit_status`.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import typer

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def _last_input_tokens(traj_path: Path) -> tuple[int | None, str | None, int]:
    data = json.loads(traj_path.read_text())
    msgs = data.get("messages") or data.get("trajectory") or []
    last_tokens: int | None = None
    n_turns = 0
    for msg in msgs:
        usage = (msg.get("usage") or {}) if isinstance(msg, dict) else {}
        tokens = usage.get("input_tokens")
        if isinstance(tokens, int):
            last_tokens = tokens
            n_turns += 1
    exit_status = None
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "exit":
            exit_status = (msg.get("extra") or {}).get("exit_status")
    info = data.get("info") or {}
    if not exit_status:
        exit_status = info.get("exit_status")
    return last_tokens, exit_status, n_turns


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def _summarize(label: str, values: list[int]) -> dict:
    if not values:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": len(values),
        "mean": int(statistics.mean(values)),
        "median": int(statistics.median(values)),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": max(values),
        "min": min(values),
    }


def _print_table(rows: list[dict]) -> None:
    cols = ["label", "n", "mean", "median", "p90", "p95", "min", "max"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    typer.echo(line)
    typer.echo("-" * len(line))
    for r in rows:
        typer.echo("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main(
    runs: list[str] = typer.Argument(None, help="result dir names under results/"),
    top: int = typer.Option(5, help="show top-N heaviest instances per run"),
) -> None:
    run_dirs = [RESULTS_ROOT / r for r in runs] if runs else sorted(
        p for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.startswith("swebench")
    )
    for run_dir in run_dirs:
        typer.echo(f"\n=== {run_dir.name} ===")
        per_status: dict[str, list[int]] = defaultdict(list)
        all_tokens: list[int] = []
        per_instance: list[tuple[str, int, str | None, int]] = []
        for traj in sorted(run_dir.glob("instance_*/*.traj.json")):
            tokens, status, n_turns = _last_input_tokens(traj)
            if tokens is None:
                continue
            all_tokens.append(tokens)
            per_status[status or "Unknown"].append(tokens)
            per_instance.append((traj.parent.name, tokens, status, n_turns))
        rows = [_summarize("ALL", all_tokens)] + [
            _summarize(status, vals) for status, vals in sorted(per_status.items())
        ]
        _print_table(rows)
        if top:
            typer.echo(f"\ntop-{top} heaviest:")
            for name, tokens, status, n_turns in sorted(per_instance, key=lambda x: -x[1])[:top]:
                typer.echo(f"  {tokens:>7}  {status or '-':<20}  turns={n_turns:<3}  {name}")


if __name__ == "__main__":
    typer.run(main)
