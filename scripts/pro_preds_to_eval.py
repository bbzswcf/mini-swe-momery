#!/usr/bin/env python3
"""
Convert mini-swe-agent batch output (preds.json, dict keyed by instance_id) to the
JSON list format expected by swe_bench_pro_eval.py (--patch_path).

Usage:
  python scripts/pro_preds_to_eval.py \\
    --input results/my_run/preds.json --output results/my_run/patches.json --prefix my_run
"""

import argparse
import json
from pathlib import Path
from typing import Any


def is_minisweagent_preds(data: Any) -> bool:
    """Return True if `data` looks like mini-swe-agent's preds.json (dict keyed
    by instance_id with `model_patch` style records), as opposed to the list
    format consumed by `swe_bench_pro_eval.py`."""
    if not isinstance(data, dict) or not data:
        return False
    sample = next(iter(data.values()))
    if not isinstance(sample, dict):
        return False
    return "model_patch" in sample or "instance_id" in sample


def convert_minisweagent_preds(data: dict, prefix: str = "minisweagent") -> list[dict[str, str]]:
    """Convert mini-swe-agent preds.json dict to swe_bench_pro_eval list format."""
    out: list[dict[str, str]] = []
    for iid, row in data.items():
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "instance_id": row.get("instance_id", iid),
                "patch": row.get("model_patch") or "",
                "prefix": prefix,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to mini-swe-agent preds.json")
    parser.add_argument("--output", required=True, help="Output JSON path for swe_bench_pro_eval.py")
    parser.add_argument(
        "--prefix",
        default="minisweagent",
        help="Value for the prefix field on each patch (default: minisweagent)",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    if not is_minisweagent_preds(data):
        raise SystemExit(
            f"{args.input} does not look like a mini-swe-agent preds.json "
            "(expected a dict keyed by instance_id with 'model_patch' records)"
        )
    out = convert_minisweagent_preds(data, prefix=args.prefix)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} patch entries to {args.output}")


if __name__ == "__main__":
    main()
