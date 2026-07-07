from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from typer.models import OptionInfo

from minisweagent.memory.evaluation import analyze_experiment, write_report

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _option(value):
    return None if isinstance(value, OptionInfo) else value


@app.command()
def main(
    experiment: Path = typer.Argument(..., help="Experiment output directory containing preds.json and trajectories."),
    eval_results: Path | None = typer.Option(None, "--eval-results", help="regraded_eval_results.json path."),
    baseline_results: Path | None = typer.Option(None, "--baseline-results", help="Baseline regraded_eval_results.json path."),
    baseline_experiment: Path | None = typer.Option(
        None,
        "--baseline-experiment",
        help="Optional baseline experiment directory for token/cost/tool-call deltas.",
    ),
    chain_nodes: Path | None = typer.Option(None, "--chain-nodes", help="Optional chain nodes JSONL for chain_id/step_index."),
    dataset: Path | None = typer.Option(
        None,
        "--dataset",
        help="Optional SWE-bench Pro JSON/JSONL with patch/problem_statement/requirements/interface fields.",
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory. Defaults to EXPERIMENT/memory_eval."),
) -> None:
    eval_results = _option(eval_results)
    baseline_results = _option(baseline_results)
    baseline_experiment = _option(baseline_experiment)
    chain_nodes = _option(chain_nodes)
    dataset = _option(dataset)
    output = _option(output)
    report = analyze_experiment(
        experiment,
        eval_results_path=eval_results,
        baseline_results_path=baseline_results,
        baseline_experiment_dir=baseline_experiment,
        chain_nodes_path=chain_nodes,
        dataset_path=dataset,
    )
    write_report(report, output or experiment / "memory_eval")
    accuracy = report.summary["accuracy"]
    memory = report.summary["memory"]
    Console().print(
        f"Wrote memory eval for [bold]{report.summary['total']}[/bold] instances: "
        f"{accuracy['passed']} passed ({accuracy['pass_rate']} pass rate), "
        f"{memory['used_instances']} used memory, "
        f"{memory['recall_successes']} successful recalls."
    )


if __name__ == "__main__":
    app()
