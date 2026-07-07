import json
from pathlib import Path

from minisweagent.memory.evaluation import analyze_experiment
from minisweagent.run.utilities.memory_eval import main


def _write_traj(
    path: Path,
    instance_id: str,
    actions: list[list[dict]],
    *,
    outputs: list[list[dict]] | None = None,
    api_calls: int = 3,
    cost: float = 1.5,
    usage: list[dict] | None = None,
    patch: str = "diff --git a/a.py b/a.py\n",
) -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    outputs = outputs or [[{"output": "ok", "returncode": 0}] for _ in actions]
    usage = usage or [{"input_tokens": 100, "output_tokens": 20, "total_tokens": 120} for _ in actions]
    for step, step_actions in enumerate(actions, start=1):
        normalized_actions = []
        for idx, action in enumerate(step_actions, start=1):
            normalized = dict(action)
            normalized.setdefault("tool_call_id", f"call_{step}_{idx}")
            normalized_actions.append(normalized)
        messages.append(
            {
                "role": "assistant",
                "content": f"thought {step}",
                "usage": usage[min(step - 1, len(usage) - 1)],
                "extra": {"actions": normalized_actions},
            }
        )
        for action, output in zip(normalized_actions, outputs[step - 1]):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": action["tool_call_id"],
                    "content": json.dumps(output),
                    "extra": {
                        "raw_output": output.get("output", ""),
                        "returncode": output.get("returncode", 0),
                    },
                }
            )
    messages.append({"role": "exit", "content": patch, "extra": {"exit_status": "Submitted", "submission": patch}})
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "info": {"model_stats": {"api_calls": api_calls, "instance_cost": cost}, "exit_status": "Submitted"},
                "instance_id": instance_id,
                "messages": messages,
                "trajectory_format": "mini-swe-agent-1.1",
            }
        )
    )


def _write_preds(path: Path, patches: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({iid: {"model_patch": patch} for iid, patch in patches.items()}))


def _write_dataset(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows))


def test_analyze_experiment_reports_accuracy_resources_and_memory_effects(tmp_path):
    experiment = tmp_path / "experiment"
    eval_results = tmp_path / "eval.json"
    baseline_results = tmp_path / "baseline.json"
    _write_preds(
        experiment / "preds.json",
        {
            "instance_repo__proj-1": "diff --git a/a.py b/a.py\n",
            "instance_repo__proj-2": "",
        },
    )
    _write_traj(
        experiment / "instance_repo__proj-1" / "instance_repo__proj-1.traj.json",
        "instance_repo__proj-1",
        [
            [{"tool_name": "session_search", "args": {"query": "parser"}}],
            [{"tool_name": "memory", "args": {"action": "add", "content": "pytest uses PYTHONPATH=."}}],
            [{"tool_name": "bash", "args": {"command": "pytest tests/test_parser.py"}}],
        ],
        outputs=[
            [
                {
                    "output": json.dumps(
                        {
                            "success": True,
                            "session_count": 2,
                            "sessions": [{"session_id": "past-1"}, {"session_id": "past-2"}],
                        }
                    ),
                    "returncode": 0,
                }
            ],
            [{"output": json.dumps({"success": True, "entries": ["pytest uses PYTHONPATH=."]}), "returncode": 0}],
            [{"output": "ok", "returncode": 0}],
        ],
        api_calls=4,
        cost=2.5,
        usage=[
            {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            {"input_tokens": 150, "output_tokens": 30, "total_tokens": 180},
            {"input_tokens": 200, "output_tokens": 40, "total_tokens": 240},
        ],
    )
    _write_traj(
        experiment / "instance_repo__proj-2" / "instance_repo__proj-2.traj.json",
        "instance_repo__proj-2",
        [[{"tool_name": "bash", "args": {"command": "pytest"}}]],
        api_calls=2,
        cost=1.0,
        patch="",
    )
    eval_results.write_text(json.dumps({"instance_repo__proj-1": True, "instance_repo__proj-2": False}))
    baseline_results.write_text(json.dumps({"instance_repo__proj-1": False, "instance_repo__proj-2": True}))

    report = analyze_experiment(experiment, eval_results_path=eval_results, baseline_results_path=baseline_results)

    assert report.summary["accuracy"] == {"evaluated": 2, "passed": 1, "failed": 1, "pass_rate": 0.5}
    assert report.summary["resources"]["api_calls"] == 6
    assert report.summary["resources"]["total_tokens"] == 660
    assert report.summary["resources"]["avg_last_response_total_tokens"] == 180.0
    assert report.summary["tools"]["bash_calls"] == 2
    assert report.summary["patches"]["empty"] == 1
    assert report.summary["memory"]["used_instances"] == 1
    assert report.summary["memory"]["recall_attempts"] == 1
    assert report.summary["memory"]["recall_successes"] == 1
    assert report.summary["memory"]["write_successes"] == 1
    assert report.summary["paired_vs_baseline"] == {"both_failed": 0, "both_passed": 0, "current_only": 1, "baseline_only": 1}
    assert report.instances["instance_repo__proj-1"].memory["returned_session_ids"] == ["past-1", "past-2"]
    assert report.instances["instance_repo__proj-1"].memory["bash_calls_before_first_recall"] == 0
    assert report.instances["instance_repo__proj-1"].process["total_tokens"] == 540
    assert report.instances["instance_repo__proj-1"].process["last_response_total_tokens"] == 240
    assert report.instances["instance_repo__proj-1"].comparison_label == "current_only_memory_recalled"
    assert report.instances["instance_repo__proj-2"].comparison_label == "baseline_only_no_memory"


def test_analyze_experiment_counts_filesystem_memory_bash_reads_and_searches(tmp_path):
    experiment = tmp_path / "experiment"
    eval_results = tmp_path / "eval.json"
    instance_id = "instance_repo__proj-1"
    _write_preds(experiment / "preds.json", {instance_id: "diff --git a/a.py b/a.py\n"})
    _write_traj(
        experiment / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [
            [
                {
                    "tool_name": "bash",
                    "args": {
                        "command": (
                            "MEMORY_CHAIN_DIR=/tmp/memory/fs/chains/chain-a && "
                            "sed -n '1,120p' \"$MEMORY_CHAIN_DIR/INDEX.md\""
                        )
                    },
                }
            ],
            [
                {
                    "tool_name": "bash",
                    "args": {
                        "command": (
                            "MEMORY_CHAIN_DIR=/tmp/memory/fs/chains/chain-a && cd \"$MEMORY_CHAIN_DIR\" "
                            "&& sed -n '1,160p' INDEX.md && rg -n \"parser\" repo.md cases || true"
                        )
                    },
                }
            ],
            [
                {
                    "tool_name": "bash",
                    "args": {
                        "command": (
                            "MEMORY_CHAIN_DIR=/tmp/memory/fs/chains/chain-a; cd /app "
                            "&& printf 'Repo root: '; pwd && ls -la"
                        )
                    },
                }
            ],
        ],
        outputs=[
            [{"output": "cases/old/summary.md: parser fix lives in src/a.py", "returncode": 0}],
            [
                {
                    "output": "repo.md:12: parser fix lives in src/a.py\ncases/old/summary.md: parser regression",
                    "returncode": 0,
                }
            ],
            [{"output": "Repo root: /app\n", "returncode": 0}],
        ],
    )
    eval_results.write_text(json.dumps({instance_id: True}))

    report = analyze_experiment(experiment, eval_results_path=eval_results)

    memory = report.instances[instance_id].memory
    assert memory["used"] is True
    assert memory["tools"] == {"filesystem_memory_read": 1, "filesystem_memory_search": 1}
    assert memory["recall_attempts"] == 2
    assert memory["recall_successes"] == 2
    assert memory["bash_calls_before_first_recall"] == 0
    assert report.summary["memory"]["used_instances"] == 1
    assert report.summary["memory"]["tools"] == {"filesystem_memory_read": 1, "filesystem_memory_search": 1}


def test_analyze_experiment_compares_baseline_trajectory_resources(tmp_path):
    experiment = tmp_path / "experiment"
    baseline = tmp_path / "baseline"
    eval_results = tmp_path / "eval.json"
    baseline_results = tmp_path / "baseline.json"
    instance_id = "instance_repo__proj-1"
    _write_preds(experiment / "preds.json", {instance_id: "diff --git a/a.py b/a.py\n"})
    _write_preds(baseline / "preds.json", {instance_id: "diff --git a/a.py b/a.py\n"})
    _write_traj(
        experiment / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [[{"tool_name": "session_search", "args": {"query": "parser"}}], [{"tool_name": "bash", "args": {"command": "pytest"}}]],
        outputs=[
            [{"output": json.dumps({"success": True, "session_count": 1, "sessions": [{"session_id": "past"}]}), "returncode": 0}],
            [{"output": "ok", "returncode": 0}],
        ],
        api_calls=2,
        cost=1.0,
        usage=[
            {"input_tokens": 80, "output_tokens": 10, "total_tokens": 90},
            {"input_tokens": 90, "output_tokens": 20, "total_tokens": 110},
        ],
    )
    _write_traj(
        baseline / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [[{"tool_name": "bash", "args": {"command": "pytest"}}]],
        api_calls=1,
        cost=0.4,
        usage=[{"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}],
    )
    eval_results.write_text(json.dumps({instance_id: True}))
    baseline_results.write_text(json.dumps({instance_id: True}))

    report = analyze_experiment(
        experiment,
        eval_results_path=eval_results,
        baseline_results_path=baseline_results,
        baseline_experiment_dir=baseline,
    )

    assert report.instances[instance_id].process["baseline"]["total_tokens"] == 60
    assert report.instances[instance_id].process["delta_vs_baseline"]["total_tokens"] == 140
    assert report.summary["resource_delta_vs_baseline"]["paired_instances"] == 1
    assert report.summary["resource_delta_vs_baseline"]["total_tokens"] == 140
    assert report.summary["resource_delta_vs_baseline"]["api_calls"] == 1


def test_analyze_experiment_classifies_provider_tools_and_output_failures(tmp_path):
    experiment = tmp_path / "experiment"
    eval_results = tmp_path / "eval.json"
    instance_id = "instance_repo__proj-1"
    _write_preds(experiment / "preds.json", {instance_id: "diff --git a/a.py b/a.py\n"})
    _write_traj(
        experiment / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [
            [{"tool_name": "mem0_search", "args": {"query": "parser"}}],
            [{"tool_name": "mem0_note", "args": {"fact": "pytest needs PYTHONPATH=."}}],
            [{"tool_name": "mem0_observe", "args": {"content": "test failure summary"}}],
        ],
        outputs=[
            [{"output": json.dumps({"success": True, "results": []}), "returncode": 0}],
            [{"output": json.dumps({"success": True}), "returncode": 0}],
            [{"output": json.dumps({"success": False, "error": "provider down"}), "returncode": 0}],
        ],
    )
    eval_results.write_text(json.dumps({instance_id: True}))

    report = analyze_experiment(experiment, eval_results_path=eval_results)

    assert report.summary["memory"]["categories"] == {"search": 1, "write": 2}
    assert report.summary["memory"]["recall_attempts"] == 1
    assert report.summary["memory"]["recall_empty"] == 1
    assert report.summary["memory"]["write_attempts"] == 2
    assert report.summary["memory"]["write_successes"] == 1
    assert report.summary["memory"]["tool_errors"] == 1


def test_analyze_experiment_counts_response_model_actions_and_outputs(tmp_path):
    experiment = tmp_path / "experiment"
    eval_results = tmp_path / "eval.json"
    instance_id = "instance_repo__proj-1"
    _write_preds(experiment / "preds.json", {instance_id: "diff --git a/a.py b/a.py\n"})
    traj_path = experiment / instance_id / f"{instance_id}.traj.json"
    traj_path.parent.mkdir()
    traj_path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "info": {"model_stats": {"api_calls": 1, "instance_cost": 0.1}},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "task"},
                    {
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "input_tokens_details": {"cached_tokens": 3},
                            "output_tokens_details": {"reasoning_tokens": 2},
                        },
                        "output": [
                            {"type": "function_call", "name": "hindsight_recall", "call_id": "call_1", "arguments": "{}"}
                        ],
                        "extra": {
                            "actions": [
                                {
                                    "tool_name": "hindsight_recall",
                                    "args": {"query": "prior parser fix"},
                                    "tool_call_id": "call_1",
                                }
                            ]
                        },
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "<returncode>0</returncode>\n<output>\n"
                        + json.dumps({"success": True, "memories": [{"id": "m1"}]})
                        + "</output>",
                    },
                ],
            }
        )
    )
    eval_results.write_text(json.dumps({instance_id: True}))

    report = analyze_experiment(experiment, eval_results_path=eval_results)

    assert report.summary["resources"]["total_tokens"] == 15
    assert report.summary["resources"]["cached_tokens"] == 3
    assert report.summary["resources"]["reasoning_tokens"] == 2
    assert report.summary["memory"]["recall_successes"] == 1
    assert report.instances[instance_id].memory["first_successful_recall_step"] == 1


def test_memory_eval_cli_writes_report_files(tmp_path):
    experiment = tmp_path / "experiment"
    eval_results = tmp_path / "eval.json"
    instance_id = "instance_repo__proj-1"
    _write_preds(experiment / "preds.json", {instance_id: "diff --git a/a.py b/a.py\n"})
    _write_traj(
        experiment / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [[{"tool_name": "memory_fs_search", "args": {"query": "parser"}}]],
        outputs=[[{"output": json.dumps({"success": True, "results": [{"path": "x"}]}), "returncode": 0}]],
    )
    eval_results.write_text(json.dumps({instance_id: True}))

    main(experiment=experiment, eval_results=eval_results)

    report = json.loads((experiment / "memory_eval" / "memory_eval_report.json").read_text())
    lines = (experiment / "memory_eval" / "instance_metrics.jsonl").read_text().splitlines()
    assert report["summary"]["memory"]["tools"] == {"memory_fs_search": 1}
    assert len(lines) == 1


def test_analyze_experiment_separates_repo_chain_and_step_breakdowns(tmp_path):
    experiment = tmp_path / "experiment"
    eval_results = tmp_path / "eval.json"
    baseline_results = tmp_path / "baseline.json"
    chain_nodes = tmp_path / "chains.jsonl"
    _write_preds(
        experiment / "preds.json",
        {
            "instance_owner__repo-a": "diff --git a/a.py b/a.py\n",
            "instance_owner__repo-b": "diff --git a/b.py b/b.py\n",
        },
    )
    _write_traj(
        experiment / "instance_owner__repo-a" / "instance_owner__repo-a.traj.json",
        "instance_owner__repo-a",
        [[{"tool_name": "memory_fs_read", "args": {"path": "repos/owner__repo/modules/api/INDEX.md"}}]],
        outputs=[[{"output": json.dumps({"success": True, "content": "prior fix"}), "returncode": 0}]],
    )
    _write_traj(
        experiment / "instance_owner__repo-b" / "instance_owner__repo-b.traj.json",
        "instance_owner__repo-b",
        [[{"tool_name": "session_search", "args": {"query": "api"}}]],
        outputs=[[{"output": json.dumps({"success": True, "session_count": 0, "sessions": []}), "returncode": 0}]],
    )
    eval_results.write_text(json.dumps({"instance_owner__repo-a": True, "instance_owner__repo-b": False}))
    baseline_results.write_text(json.dumps({"instance_owner__repo-a": False, "instance_owner__repo-b": True}))
    chain_nodes.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "instance_owner__repo-a",
                        "repo": "owner/repo",
                        "chain_id": "owner__repo-0001",
                        "step_index": 1,
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "instance_owner__repo-b",
                        "repo": "owner/repo",
                        "chain_id": "owner__repo-0001",
                        "step_index": 18,
                    }
                ),
            ]
        )
    )

    report = analyze_experiment(
        experiment,
        eval_results_path=eval_results,
        baseline_results_path=baseline_results,
        chain_nodes_path=chain_nodes,
    )

    assert report.instances["instance_owner__repo-a"].outcome["repo"] == "owner/repo"
    assert report.instances["instance_owner__repo-a"].outcome["chain_id"] == "owner__repo-0001"
    assert report.instances["instance_owner__repo-b"].outcome["step_bucket"] == "16+"
    assert report.summary["repo_breakdown"]["owner/repo"]["total"] == 2
    assert report.summary["chain_breakdown"]["owner__repo-0001"]["total"] == 2
    assert report.summary["step_breakdown"]["16+"] == {
        "total": 1,
        "passed": 0,
        "pass_rate": 0.0,
        "baseline_passed": 1,
        "memory_used": 1,
        "current_only": 0,
        "baseline_only": 1,
    }


def test_patch_localized_exploration_and_memory_influence_metrics_use_gold_patch_surface(tmp_path):
    experiment = tmp_path / "experiment"
    dataset = tmp_path / "swe_bench_pro.json"
    eval_results = tmp_path / "eval.json"
    instance_id = "instance_repo__proj-1"
    gold_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -10,6 +10,7 @@ def public_func():
 old_value = helper()
-value = helper()
+value = helper_fix()
 return value
@@ -100,3 +101,4 @@ def other():
-old2 = 1
+old2 = 2
"""
    model_patch = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -14,3 +14,3 @@ def public_func():
-value = helper()
+value = helper_fix()
diff --git a/src/extra.py b/src/extra.py
--- a/src/extra.py
+++ b/src/extra.py
@@ -50,2 +50,2 @@
-x = 1
+x = 2
"""
    _write_dataset(
        dataset,
        [
            {
                "instance_id": instance_id,
                "patch": gold_patch,
                "problem_statement": "Fix public_func behavior.",
                "requirements": "The implementation must keep public_func compatible.",
                "interface": "Name: public_func\nPath: src/a.py\nDescription: public interface.",
            }
        ],
    )
    _write_preds(experiment / "preds.json", {instance_id: model_patch})
    baseline = tmp_path / "baseline"
    _write_preds(baseline / "preds.json", {instance_id: model_patch})
    _write_traj(
        experiment / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [
            [{"tool_name": "bash", "args": {"command": "rg helper_fix src/a.py"}}],
            [{"tool_name": "session_search", "args": {"query": "prior helper_fix fix"}}],
            [{"tool_name": "bash", "args": {"command": "sed -n '95,120p' src/a.py"}}],
            [{"tool_name": "bash", "args": {"command": "sed -n '95,120p' src/a.py"}}],
            [{"tool_name": "bash", "args": {"command": "sed -n '200,220p' src/other.py"}}],
        ],
        outputs=[
            [{"output": "src/a.py: helper_fix", "returncode": 0}],
            [
                {
                    "output": json.dumps(
                        {
                            "success": True,
                            "sessions": [
                                {
                                    "session_id": "past",
                                    "content": "Use helper_fix in src/a.py, not the older public_func path.",
                                }
                            ],
                        }
                    ),
                    "returncode": 0,
                }
            ],
            [{"output": "lines", "returncode": 0}],
            [{"output": "lines again", "returncode": 0}],
            [{"output": "other lines", "returncode": 0}],
        ],
        patch=model_patch,
    )
    _write_traj(
        baseline / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [
            [{"tool_name": "bash", "args": {"command": "rg helper_fix src/a.py"}}],
            [{"tool_name": "bash", "args": {"command": "python -m pytest tests/test_a.py"}}],
            [{"tool_name": "bash", "args": {"command": "sed -n '95,120p' src/a.py"}}],
        ],
        outputs=[
            [{"output": "src/a.py: helper_fix", "returncode": 0}],
            [{"output": "failed", "returncode": 1}],
            [{"output": "lines", "returncode": 0}],
        ],
        patch=model_patch,
    )
    eval_results.write_text(json.dumps({instance_id: True}))

    report = analyze_experiment(
        experiment,
        eval_results_path=eval_results,
        baseline_experiment_dir=baseline,
        dataset_path=dataset,
    )
    metrics = report.instances[instance_id]

    assert metrics.patch_alignment["edit_hunk_recall_w10"] == 0.5
    assert metrics.patch_alignment["edit_hunk_precision_w10"] == 0.5
    assert metrics.patch_alignment["edit_hunk_f1_w10"] == 0.5
    assert metrics.localized_exploration["target_region_view_recall_w50"] == 0.5
    assert metrics.localized_exploration["first_target_region_step"] == 3
    assert metrics.localized_exploration["auc_target_region_recall_w50"] == 0.3
    assert metrics.exploration_efficiency["line_redundancy"] == 0.3562
    assert metrics.memory_influence["recall_patch_identifier_relevance"] == 0.5
    assert metrics.memory_influence["novel_patch_signal_count"] == 1
    assert metrics.memory_influence["memory_action_follow_rate_next5"] == 1.0
    assert metrics.memory_influence["post_recall_target_region_gain_next5"] == 0.5
    assert metrics.memory_influence["baseline_expected_target_region_gain_next5"] == 0.5
    assert metrics.memory_influence["post_recall_target_region_gain_delta_vs_baseline_next5"] == 0.0
    assert report.summary["patch_alignment"]["avg_edit_hunk_f1_w10"] == 0.5
    assert report.summary["localized_exploration"]["target_region_found_rate"] == 1.0
    assert report.summary["exploration_efficiency"]["line_view_instances"] == 1
    assert report.summary["memory_influence"]["successful_recall_instances"] == 1
    assert report.summary["memory_influence"]["avg_recalled_files"] == 1.0
    assert report.summary["memory_influence"]["avg_recalled_identifiers"] == 2.0
    assert report.summary["memory_influence"]["avg_novel_patch_signal_count"] == 1.0
    assert report.summary["memory_influence"]["avg_post_recall_target_region_gain_delta_vs_baseline_next5"] == 0.0


def test_patch_alignment_counts_empty_model_patch_as_zero_f1(tmp_path):
    experiment = tmp_path / "experiment"
    dataset = tmp_path / "swe_bench_pro.json"
    instance_id = "instance_repo__proj-1"
    _write_dataset(
        dataset,
        [
            {
                "instance_id": instance_id,
                "patch": """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -10,2 +10,2 @@
-old = 1
+old = 2
""",
            }
        ],
    )
    _write_preds(experiment / "preds.json", {instance_id: ""})
    _write_traj(
        experiment / instance_id / f"{instance_id}.traj.json",
        instance_id,
        [[{"tool_name": "bash", "args": {"command": "pytest"}}]],
        patch="",
    )

    report = analyze_experiment(experiment, dataset_path=dataset)
    metrics = report.instances[instance_id].patch_alignment

    assert metrics["eligible_gold_hunks"] == 1
    assert metrics["eligible_pred_hunks"] == 0
    assert metrics["edit_hunk_recall_w10"] == 0.0
    assert metrics["edit_hunk_precision_w10"] == 0.0
    assert metrics["edit_hunk_f1_w10"] == 0.0
    assert report.summary["patch_alignment"]["avg_edit_hunk_f1_w10"] == 0.0
