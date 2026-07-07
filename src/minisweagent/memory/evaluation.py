from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MEMORY_TOOL_PREFIXES = ("hindsight_", "mem0_", "memory_fs_")
MEMORY_TOOL_NAMES = {"memory", "session_search"}
RESOURCE_KEYS = (
    "api_calls",
    "instance_cost",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "tool_calls",
    "bash_calls",
)
PATCH_ALIGNMENT_WINDOW = 10
TARGET_REGION_WINDOW = 50
MEMORY_ACTION_WINDOW = 5

_HUNK_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?")
_PATH_RE = re.compile(r"(?<![\w.-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]+")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "api",
    "behavior",
    "class",
    "compatible",
    "description",
    "fix",
    "function",
    "implementation",
    "in",
    "input",
    "interface",
    "method",
    "must",
    "name",
    "new",
    "not",
    "older",
    "on",
    "output",
    "path",
    "public",
    "return",
    "returns",
    "the",
    "to",
    "type",
    "use",
    "uses",
    "with",
}


@dataclass
class InstanceMetrics:
    instance_id: str
    passed: bool | None
    baseline_passed: bool | None
    outcome: dict
    process: dict
    memory: dict
    patch_alignment: dict
    localized_exploration: dict
    exploration_efficiency: dict
    memory_influence: dict
    comparison_label: str


@dataclass
class MemoryEvalReport:
    experiment: str
    summary: dict
    instances: dict[str, InstanceMetrics]

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment,
            "summary": self.summary,
            "instances": {k: asdict(v) for k, v in self.instances.items()},
        }


def analyze_experiment(
    experiment_dir: Path | str,
    *,
    eval_results_path: Path | str | None = None,
    baseline_results_path: Path | str | None = None,
    baseline_experiment_dir: Path | str | None = None,
    chain_nodes_path: Path | str | None = None,
    dataset_path: Path | str | None = None,
) -> MemoryEvalReport:
    experiment_dir = Path(experiment_dir)
    eval_results = _load_results(eval_results_path)
    baseline_results = _load_results(baseline_results_path)
    dataset = _load_dataset_instances(dataset_path)
    chain_nodes = _load_chain_nodes(chain_nodes_path)
    preds = _load_preds(experiment_dir / "preds.json")
    baseline_metrics = _load_baseline_metrics(baseline_experiment_dir, dataset)

    instances: dict[str, InstanceMetrics] = {}
    for traj_path in _iter_trajectory_paths(experiment_dir):
        traj = json.loads(traj_path.read_text())
        instance_id = str(traj.get("instance_id") or traj_path.parent.name)
        passed = eval_results.get(instance_id)
        baseline_passed = baseline_results.get(instance_id)
        events = _action_events(traj.get("messages") or [])
        process = _process_metrics(traj, events)
        baseline = baseline_metrics.get(instance_id) or baseline_metrics.get(instance_id.removeprefix("instance_"))
        if baseline is not None:
            process["baseline"] = baseline["process"]
            process["delta_vs_baseline"] = _resource_delta(process, baseline["process"])
        memory = _memory_metrics(events)
        chain = chain_nodes.get(instance_id) or chain_nodes.get(instance_id.removeprefix("instance_")) or {}
        pred_patch = _patch_for(preds, instance_id)
        traj_patch = str((traj.get("info") or {}).get("submission") or _last_exit_submission(traj) or "")
        patch = pred_patch if pred_patch or instance_id in preds or instance_id.removeprefix("instance_") in preds else traj_patch
        instance_data = _dataset_instance(dataset, instance_id)
        gold_patch = str(instance_data.get("patch") or instance_data.get("gold_patch") or "")
        prompt_surface = _prompt_surface(instance_data)
        trajectory = _trajectory_surface(events)
        patch_alignment = _patch_alignment_metrics(gold_patch, patch)
        localized_exploration = _localized_exploration_metrics(gold_patch, trajectory)
        exploration_efficiency = _exploration_efficiency_metrics(gold_patch, trajectory)
        memory_influence = _memory_influence_metrics(
            events,
            gold_patch,
            prompt_surface,
            trajectory,
            baseline_localized=baseline.get("localized_exploration") if baseline else None,
        )
        outcome = {
            "passed": passed,
            "empty_patch": not patch.strip(),
            "patch_files": _patch_files(patch),
            "preds_traj_patch_mismatch": bool(pred_patch and traj_patch and pred_patch != traj_patch),
            "exit_status": (traj.get("info") or {}).get("exit_status", ""),
            "repo": chain.get("repo") or _repo_from_instance(instance_id),
            "chain_id": chain.get("chain_id") or chain.get("memory_session_id") or _repo_from_instance(instance_id),
            "step_index": chain.get("step_index"),
            "step_bucket": _step_bucket(chain.get("step_index")),
        }
        instances[instance_id] = InstanceMetrics(
            instance_id=instance_id,
            passed=passed,
            baseline_passed=baseline_passed,
            outcome=outcome,
            process=process,
            memory=memory,
            patch_alignment=patch_alignment,
            localized_exploration=localized_exploration,
            exploration_efficiency=exploration_efficiency,
            memory_influence=memory_influence,
            comparison_label=_comparison_label(passed, baseline_passed, memory),
        )
    return MemoryEvalReport(
        experiment=str(experiment_dir),
        summary=_summary(instances, eval_results=eval_results, baseline_results=baseline_results),
        instances=instances,
    )


def write_report(report: MemoryEvalReport, output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "memory_eval_report.json").write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    lines = [json.dumps(asdict(i), ensure_ascii=False) for i in report.instances.values()]
    (output_dir / "instance_metrics.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))


def _iter_trajectory_paths(experiment_dir: Path) -> list[Path]:
    paths = sorted(experiment_dir.glob("instance_*/*.traj.json"))
    return paths or sorted(p for p in experiment_dir.rglob("*.traj.json") if "memory_eval" not in p.parts)


def _load_results(path: Path | str | None) -> dict[str, bool]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and all(isinstance(v, bool) for v in data.values()):
        return {str(k): bool(v) for k, v in data.items()}
    if isinstance(data, dict) and "resolved" in data:
        return {str(i): True for i in data.get("resolved", [])} | {str(i): False for i in data.get("failed", [])}
    if isinstance(data, dict) and "passed_ids" in data:
        return {str(i): True for i in data.get("passed_ids", [])} | {str(i): False for i in data.get("failed_ids", [])}
    return {}


def _load_preds(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {str(x.get("instance_id")): x for x in data if isinstance(x, dict)}
    return data if isinstance(data, dict) else {}


def _load_chain_nodes(path: Path | str | None) -> dict[str, dict]:
    if path is None:
        return {}
    nodes = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        instance_id = str(data.get("instance_id", ""))
        if not instance_id:
            continue
        nodes[instance_id] = data
        stripped = instance_id.removeprefix("instance_")
        nodes[stripped] = data
        nodes[f"instance_{stripped}"] = data
    return nodes


def _load_dataset_instances(path: Path | str | None) -> dict[str, dict]:
    if path is None:
        default = Path("data/swe_bench_pro.json")
        path = default if default.exists() else None
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else [data]
    instances: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id") or row.get("original_inst_id") or "")
        if not instance_id:
            continue
        stripped = instance_id.removeprefix("instance_")
        instances[instance_id] = row
        instances[stripped] = row
        instances[f"instance_{stripped}"] = row
    return instances


def _dataset_instance(dataset: dict[str, dict], instance_id: str) -> dict:
    return dataset.get(instance_id) or dataset.get(instance_id.removeprefix("instance_")) or {}


def _load_baseline_metrics(path: Path | str | None, dataset: dict[str, dict]) -> dict[str, dict]:
    if path is None:
        return {}
    result: dict[str, dict] = {}
    experiment_dir = Path(path)
    preds = _load_preds(experiment_dir / "preds.json")
    for traj_path in _iter_trajectory_paths(experiment_dir):
        traj = json.loads(traj_path.read_text())
        instance_id = str(traj.get("instance_id") or traj_path.parent.name)
        events = _action_events(traj.get("messages") or [])
        pred_patch = _patch_for(preds, instance_id)
        traj_patch = str((traj.get("info") or {}).get("submission") or _last_exit_submission(traj) or "")
        patch = pred_patch if pred_patch or instance_id in preds or instance_id.removeprefix("instance_") in preds else traj_patch
        instance_data = _dataset_instance(dataset, instance_id)
        gold_patch = str(instance_data.get("patch") or instance_data.get("gold_patch") or "")
        trajectory = _trajectory_surface(events)
        metrics = {
            "process": _process_metrics(traj, events, include_baseline=False),
            "patch_alignment": _patch_alignment_metrics(gold_patch, patch),
            "localized_exploration": _localized_exploration_metrics(gold_patch, trajectory),
            "exploration_efficiency": _exploration_efficiency_metrics(gold_patch, trajectory),
        }
        result[instance_id] = metrics
        stripped = instance_id.removeprefix("instance_")
        result[stripped] = metrics
        result[f"instance_{stripped}"] = metrics
    return result


def _patch_for(preds: dict[str, Any], instance_id: str) -> str:
    pred = preds.get(instance_id) or preds.get(instance_id.removeprefix("instance_")) or {}
    if isinstance(pred, dict):
        return str(pred.get("model_patch", ""))
    return str(pred)


def _last_exit_submission(traj: dict) -> str:
    for msg in reversed(traj.get("messages") or []):
        if msg.get("role") == "exit":
            return str((msg.get("extra") or {}).get("submission") or msg.get("content") or "")
    return ""


def _patch_files(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(.*?) b/", patch or "", flags=re.M)))


def _process_metrics(traj: dict, events: list[dict], *, include_baseline: bool = True) -> dict:
    messages = traj.get("messages") or []
    usage = Counter()
    last_usage: dict[str, int] = {}
    response_count = 0
    for msg in messages:
        msg_usage = _message_usage(msg)
        if msg_usage:
            response_count += 1
            usage.update(msg_usage)
            last_usage = msg_usage
    model_stats = ((traj.get("info") or {}).get("model_stats") or {})
    api_calls = int(model_stats.get("api_calls") or response_count or 0)
    process = {
        "api_calls": api_calls,
        "instance_cost": float(model_stats.get("instance_cost") or 0),
        "messages": len(messages),
        "assistant_turns": sum(1 for m in messages if m.get("role") == "assistant" or (m.get("extra") or {}).get("actions")),
        "tool_calls": len(events),
        "bash_calls": sum(1 for e in events if e["name"] == "bash"),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "cached_tokens": usage["cached_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "last_response_input_tokens": last_usage.get("input_tokens", 0),
        "last_response_output_tokens": last_usage.get("output_tokens", 0),
        "last_response_total_tokens": last_usage.get("total_tokens", 0),
    }
    if not include_baseline:
        baseline_keys = {
            "messages",
            "assistant_turns",
            "last_response_input_tokens",
            "last_response_output_tokens",
            "last_response_total_tokens",
        }
        return {k: v for k, v in process.items() if k in RESOURCE_KEYS or k in baseline_keys}
    return process


def _message_usage(msg: dict) -> dict[str, int]:
    usage = msg.get("usage")
    if not isinstance(usage, dict):
        usage = (((msg.get("extra") or {}).get("response") or {}).get("usage") or {})
    if not isinstance(usage, dict) or not usage:
        return {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": int(input_details.get("cached_tokens") or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
    }


def _action_events(messages: list[dict]) -> list[dict]:
    outputs = _tool_outputs(messages)
    output_by_id = {o["call_id"]: o for o in outputs if o.get("call_id")}
    events = []
    output_index = 0
    step = 0
    action_index = 0
    for message in messages:
        actions = (message.get("extra") or {}).get("actions") or []
        if message.get("role") == "assistant" or actions:
            step += 1
        for action in actions:
            action_index += 1
            call_id = action.get("tool_call_id") or action.get("call_id") or action.get("id")
            output = output_by_id.get(call_id) if call_id else None
            if output is None and output_index < len(outputs):
                output = outputs[output_index]
            output_index += 1
            events.append(
                {
                    "index": action_index,
                    "step": step,
                    "name": _tool_name(action),
                    "args": action.get("args") or {},
                    "call_id": call_id,
                    "output": output or {},
                }
            )
    return events


def _tool_outputs(messages: list[dict]) -> list[dict]:
    outputs: list[dict] = []
    for msg in messages:
        if msg.get("type") == "function_call_output":
            outputs.append(_normalize_tool_output(msg, call_id=msg.get("call_id")))
        elif msg.get("role") == "tool":
            call_id = msg.get("tool_call_id") or msg.get("call_id")
            raw = (msg.get("extra") or {}).get("raw_output")
            content = msg.get("content")
            parsed = _json_loads_maybe(content)
            if call_id or not isinstance(parsed, list):
                outputs.append(_normalize_tool_output(msg, call_id=call_id, raw=raw if raw is not None else content))
            else:
                for item in parsed:
                    outputs.append(_normalize_tool_output({}, raw=json.dumps(item), parsed=item))
    return outputs


def _normalize_tool_output(
    msg: dict,
    *,
    call_id: str | None = None,
    raw: Any | None = None,
    parsed: Any | None = None,
) -> dict:
    if raw is None:
        raw = (msg.get("extra") or {}).get("raw_output")
    if raw is None:
        raw = msg.get("output")
    if raw is None:
        raw = msg.get("content")
    raw = str(raw or "")
    parsed = parsed if parsed is not None else _parse_output_payload(raw)
    returncode = (msg.get("extra") or {}).get("returncode")
    if isinstance(parsed, dict) and "returncode" in parsed:
        returncode = parsed.get("returncode")
    nested = parsed
    if isinstance(parsed, dict) and "output" in parsed:
        nested = _parse_output_payload(str(parsed.get("output") or ""))
    return {
        "call_id": call_id,
        "raw": raw,
        "returncode": returncode,
        "data": nested,
        "success": _output_success(nested, returncode),
    }


def _parse_output_payload(raw: str) -> Any:
    raw = str(raw or "").strip()
    match = re.search(r"<output>\s*(.*?)</output>", raw, flags=re.S)
    if match:
        raw = match.group(1).strip()
    return _json_loads_maybe(raw)


def _json_loads_maybe(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def _output_success(data: Any, returncode: Any) -> bool | None:
    if isinstance(data, dict) and isinstance(data.get("success"), bool):
        return bool(data["success"])
    if isinstance(data, dict) and data.get("error"):
        return False
    if returncode is None:
        return None
    try:
        return int(returncode) == 0
    except Exception:
        return None


def _memory_metrics(events: list[dict]) -> dict:
    memory_events = _memory_events(events)
    tools = Counter(event["name"] for event in memory_events)
    categories = Counter(_memory_category(event["name"]) for event in memory_events)
    first_memory = memory_events[0] if memory_events else None
    recall_events = [event for event in memory_events if _is_memory_recall_event(event)]
    write_events = [event for event in memory_events if _memory_category(event["name"]) == "write"]
    returned_session_ids: list[str] = []
    recall_successes = 0
    recall_empty = 0
    first_successful_recall = None
    tool_errors = 0
    write_successes = 0

    for event in memory_events:
        output = event.get("output") or {}
        success = output.get("success")
        if success is False:
            tool_errors += 1
        if _is_memory_recall_event(event):
            result_count = _memory_result_count(output.get("data"))
            if success is False:
                continue
            if result_count > 0:
                recall_successes += 1
                first_successful_recall = first_successful_recall or event
            else:
                recall_empty += 1
            returned_session_ids.extend(_returned_session_ids(output.get("data")))
        elif _memory_category(event["name"]) == "write" and success is not False:
            write_successes += 1

    first_recall = recall_events[0] if recall_events else None
    first_recall_index = first_recall["index"] if first_recall else None
    return {
        "used": bool(memory_events),
        "tool_calls": len(memory_events),
        "first_memory_step": first_memory["step"] if first_memory else None,
        "first_recall_step": first_recall["step"] if first_recall else None,
        "first_successful_recall_step": first_successful_recall["step"] if first_successful_recall else None,
        "tools": dict(sorted(tools.items())),
        "categories": dict(sorted(categories.items())),
        "recall_attempts": len(recall_events),
        "recall_successes": recall_successes,
        "recall_empty": recall_empty,
        "write_attempts": len(write_events),
        "write_successes": write_successes,
        "tool_errors": tool_errors,
        "returned_session_ids": sorted(set(returned_session_ids)),
        "bash_calls_before_first_recall": (
            sum(1 for e in events if e["name"] == "bash" and e["index"] < first_recall_index)
            if first_recall_index is not None
            else None
        ),
        "bash_calls_after_first_recall": (
            sum(1 for e in events if e["name"] == "bash" and e["index"] > first_recall_index)
            if first_recall_index is not None
            else None
        ),
    }


def _memory_events(events: list[dict]) -> list[dict]:
    memory_events = []
    for event in events:
        if _is_memory_tool(event["name"]):
            memory_events.append(event)
            continue
        filesystem_event = _filesystem_memory_event(event)
        if filesystem_event is not None:
            memory_events.append(filesystem_event)
    return memory_events


def _filesystem_memory_event(event: dict) -> dict | None:
    if event.get("name") != "bash":
        return None
    command = str((event.get("args") or {}).get("command") or "")
    if not _uses_filesystem_memory(command):
        return None
    output = dict(event.get("output") or {})
    data = output.get("data")
    content = _filesystem_memory_output_text(data if data is not None else output.get("raw"))
    output["data"] = {"content": content} if content else {}
    return {
        **event,
        "name": "filesystem_memory_search" if _filesystem_memory_command_is_search(command) else "filesystem_memory_read",
        "output": output,
    }


def _uses_filesystem_memory(command: str) -> bool:
    command = str(command or "")
    without_assignment = re.sub(
        r"\bMEMORY_CHAIN_DIR=(?:'[^']*'|\"[^\"]*\"|[^\s;&|]+)",
        "MEMORY_CHAIN_DIR=",
        command,
    )
    if re.search(r"\bcd\s+(?:\"?\$MEMORY_CHAIN_DIR\"?|'?\$MEMORY_CHAIN_DIR'?|\"?\$\{MEMORY_CHAIN_DIR\}\"?)", without_assignment):
        return True
    if re.search(r"\$(?:\{MEMORY_CHAIN_DIR\}|MEMORY_CHAIN_DIR)/", without_assignment):
        return True
    if re.search(r"(?:^|[^\w.-])(?:chain_memory|fs/chains)/", without_assignment):
        return True
    if "MEMORY_CHAIN_DIR" not in without_assignment:
        return False
    return bool(
        re.search(r"\b(?:README|INDEX|repo)\.md\b", without_assignment)
        or re.search(r"\bcases/[^\s;&|]+/(?:summary|trajectory|task)\.md\b", without_assignment)
    )


def _filesystem_memory_command_is_search(command: str) -> bool:
    return bool(re.search(r"\b(?:rg|grep|ag|ack)\b", command or ""))


def _filesystem_memory_output_text(data: Any) -> str:
    text = _memory_output_text(data)
    text = text.strip()
    if not text:
        return ""
    empty_markers = {"no repo.md", "no index.md", "no memory", "not found"}
    if text.lower() in empty_markers:
        return ""
    return text


def _memory_result_count(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    if isinstance(data.get("count"), int):
        return int(data["count"])
    for key in ("sessions", "results", "memories", "matches"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    if isinstance(data.get("session_count"), int):
        return int(data["session_count"])
    if data.get("content") or data.get("answer"):
        return 1
    return 0


def _returned_session_ids(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return []
    sessions = data.get("sessions")
    if not isinstance(sessions, list):
        return []
    return [str(s.get("session_id")) for s in sessions if isinstance(s, dict) and s.get("session_id")]


def _tool_name(action: dict) -> str:
    return str(action.get("tool_name") or "bash")


def _is_memory_tool(name: str) -> bool:
    return name in MEMORY_TOOL_NAMES or name.startswith(MEMORY_TOOL_PREFIXES)


def _memory_category(name: str) -> str:
    if name == "session_search" or name.endswith("_search") or name.endswith("_recall") or name.endswith("_reflect"):
        return "search"
    if name.endswith("_read"):
        return "read"
    if name.endswith("_tree") or name.endswith("_list"):
        return "tree"
    if name.endswith("_write") or name in {"memory", "hindsight_retain", "mem0_add", "mem0_note", "mem0_observe"}:
        return "write"
    return "other"


def _is_memory_recall_event(event: dict) -> bool:
    name = event.get("name") or ""
    return _memory_category(str(name)) == "search" or name == "filesystem_memory_read"


def _comparison_label(passed: bool | None, baseline_passed: bool | None, memory: dict) -> str:
    if baseline_passed is None or passed is None:
        prefix = "unpaired"
    elif passed and not baseline_passed:
        prefix = "current_only"
    elif not passed and baseline_passed:
        prefix = "baseline_only"
    elif passed and baseline_passed:
        prefix = "both_passed"
    else:
        prefix = "both_failed"
    if memory.get("recall_successes", 0) > 0:
        suffix = "memory_recalled"
    elif memory.get("used"):
        suffix = "memory_used"
    else:
        suffix = "no_memory"
    return f"{prefix}_{suffix}"


def _parse_patch_hunks(patch: str) -> list[dict]:
    hunks: list[dict] = []
    current_file = ""
    current_hunk: dict | None = None
    for line in (patch or "").splitlines():
        diff_match = re.match(r"^diff --git a/(.*?) b/(.*?)$", line)
        if diff_match:
            current_file = diff_match.group(2)
            current_hunk = None
            continue
        hunk_match = _HUNK_RE.match(line)
        if hunk_match and current_file:
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2) or 1)
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4) or 1)
            current_hunk = {
                "file": current_file,
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "changed_text": [],
            }
            hunks.append(current_hunk)
            continue
        if current_hunk is not None and (line.startswith("+") or line.startswith("-")):
            if not line.startswith(("+++", "---")):
                current_hunk["changed_text"].append(line[1:])
    return hunks


def _eligible_hunks(hunks: list[dict]) -> list[dict]:
    return [h for h in hunks if int(h.get("old_count") or 0) > 0 and h.get("file")]


def _hunk_interval(hunk: dict, window: int = 0) -> tuple[int, int]:
    start = max(1, int(hunk["old_start"]) - window)
    end = int(hunk["old_start"]) + max(1, int(hunk["old_count"])) - 1 + window
    return start, end


def _patch_alignment_metrics(gold_patch: str, model_patch: str) -> dict:
    gold_hunks_all = _parse_patch_hunks(gold_patch)
    pred_hunks_all = _parse_patch_hunks(model_patch)
    gold_hunks = _eligible_hunks(gold_hunks_all)
    pred_hunks = _eligible_hunks(pred_hunks_all)
    hit_gold: set[int] = set()
    hit_pred: set[int] = set()
    for gi, gold in enumerate(gold_hunks):
        gold_interval = _hunk_interval(gold, PATCH_ALIGNMENT_WINDOW)
        for pi, pred in enumerate(pred_hunks):
            if gold["file"] == pred["file"] and _intervals_overlap(gold_interval, _hunk_interval(pred)):
                hit_gold.add(gi)
                hit_pred.add(pi)
    recall = _safe_div(len(hit_gold), len(gold_hunks))
    precision = _safe_div(len(hit_pred), len(pred_hunks))
    if gold_hunks and not pred_hunks:
        precision = 0.0
    return {
        "eligible": bool(gold_hunks),
        "window": PATCH_ALIGNMENT_WINDOW,
        "eligible_gold_hunks": len(gold_hunks),
        "eligible_pred_hunks": len(pred_hunks),
        "new_file_gold_hunks": len(gold_hunks_all) - len(gold_hunks),
        "new_file_pred_hunks": len(pred_hunks_all) - len(pred_hunks),
        "edit_hunk_recall_w10": _round_or_none(recall),
        "edit_hunk_precision_w10": _round_or_none(precision),
        "edit_hunk_f1_w10": _round_or_none(_f1(recall, precision)),
    }


def _trajectory_surface(events: list[dict]) -> dict:
    files_by_step: dict[int, set[str]] = {}
    lines_by_step: dict[int, dict[str, list[tuple[int, int]]]] = {}
    searches: list[str] = []
    for event in events:
        step = int(event.get("step") or 0)
        if step <= 0:
            continue
        files: set[str] = set()
        ranges: dict[str, list[tuple[int, int]]] = {}
        if event.get("name") == "bash":
            command = str((event.get("args") or {}).get("command") or "")
            files.update(_extract_files_from_text(command))
            for file_path, start, end in _view_ranges_from_command(command):
                files.add(file_path)
                ranges.setdefault(file_path, []).append((start, end))
            normalized_search = _normalize_search_command(command)
            if normalized_search:
                searches.append(normalized_search)
        else:
            args_text = json.dumps(event.get("args") or {}, ensure_ascii=False)
            files.update(_extract_files_from_text(args_text))
        if files:
            files_by_step.setdefault(step, set()).update(files)
        for file_path, intervals in ranges.items():
            lines_by_step.setdefault(step, {}).setdefault(file_path, []).extend(intervals)
    return {
        "total_steps": max((event.get("step") or 0 for event in events), default=0),
        "files_by_step": {step: sorted(files) for step, files in files_by_step.items()},
        "lines_by_step": {
            step: {file_path: _merge_intervals(intervals) for file_path, intervals in by_file.items()}
            for step, by_file in lines_by_step.items()
        },
        "searches": searches,
    }


def _view_ranges_from_command(command: str) -> list[tuple[str, int, int]]:
    views: list[tuple[str, int, int]] = []
    for match in re.finditer(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+([^\s&|>;<]+)", command):
        views.append((_normalize_path(match.group(3)), int(match.group(1)), int(match.group(2))))
    for match in re.finditer(r"nl\s+[^|]+\s+([^\s|]+)\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p", command):
        views.append((_normalize_path(match.group(1)), int(match.group(2)), int(match.group(3))))
    for match in re.finditer(r"\bhead\s+-n\s+(\d+)\s+([^\s&|>;<]+)", command):
        views.append((_normalize_path(match.group(2)), 1, int(match.group(1))))
    return [(file_path, min(start, end), max(start, end)) for file_path, start, end in views if file_path]


def _normalize_search_command(command: str) -> str:
    command = " ".join((command or "").split())
    match = re.search(r"\b(rg|grep)\b\s+(?:-[^\s]+\s+)*['\"]?([^'\"\s]+)['\"]?(?:\s+([^\s&|;]+))?", command)
    if not match:
        return ""
    tool, query, scope = match.group(1), match.group(2), match.group(3) or "."
    return f"{tool}:{query}:{_normalize_path(scope)}"


def _localized_exploration_metrics(gold_patch: str, trajectory: dict) -> dict:
    target_regions = _target_regions(gold_patch, TARGET_REGION_WINDOW)
    eligible = len(target_regions)
    total_steps = int(trajectory.get("total_steps") or 0)
    lines_by_step = trajectory.get("lines_by_step") or {}
    cumulative: dict[str, list[tuple[int, int]]] = {}
    hit_hunks: set[int] = set()
    recall_by_step: list[float] = []
    first_target_step = None
    for step in range(1, total_steps + 1):
        for file_path, intervals in lines_by_step.get(step, {}).items():
            cumulative[file_path] = _merge_intervals(cumulative.get(file_path, []) + intervals)
        before = len(hit_hunks)
        hit_hunks.update(_hit_target_region_indexes(cumulative, target_regions))
        if first_target_step is None and len(hit_hunks) > before:
            first_target_step = step
        recall_by_step.append((len(hit_hunks) / eligible) if eligible else 0.0)

    final_recall = _safe_div(len(hit_hunks), eligible)
    target_by_file = _regions_by_file(target_regions)
    viewed_unique = _union_lines(lines_by_step)
    viewed_total = _line_total(viewed_unique)
    overlap = _line_intersection_total(viewed_unique, target_by_file)
    precision = _safe_div(overlap, viewed_total)
    auc = sum(recall_by_step) / len(recall_by_step) if recall_by_step else (1.0 if not eligible else 0.0)
    return {
        "eligible": bool(eligible),
        "window": TARGET_REGION_WINDOW,
        "eligible_gold_hunks": eligible,
        "target_region_view_recall_w50": _round_or_none(final_recall),
        "target_region_view_precision_w50": _round_or_none(precision),
        "first_target_region_step": first_target_step,
        "target_region_found": first_target_step is not None,
        "auc_target_region_recall_w50": round(auc, 4),
        "target_region_recall_by_step": [round(value, 4) for value in recall_by_step],
        "explicit_line_view_steps": len(lines_by_step),
    }


def _exploration_efficiency_metrics(gold_patch: str, trajectory: dict) -> dict:
    lines_by_step = trajectory.get("lines_by_step") or {}
    target_by_file = _regions_by_file(_target_regions(gold_patch, TARGET_REGION_WINDOW))
    per_step_total = sum(_line_total(by_file) for by_file in lines_by_step.values())
    unique_total = _line_total(_union_lines(lines_by_step))
    off_target_by_step = {
        step: _subtract_regions(by_file, target_by_file) for step, by_file in lines_by_step.items()
    }
    off_per_step_total = sum(_line_total(by_file) for by_file in off_target_by_step.values())
    off_unique_total = _line_total(_union_lines(off_target_by_step))
    searches = trajectory.get("searches") or []
    return {
        "line_redundancy": _round_or_none(_redundancy(unique_total, per_step_total)),
        "off_target_line_redundancy": _round_or_none(_redundancy(off_unique_total, off_per_step_total)),
        "search_redundancy": _round_or_none(_redundancy(len(set(searches)), len(searches))),
        "repeated_searches": len(searches) - len(set(searches)),
    }


def _memory_influence_metrics(
    events: list[dict],
    gold_patch: str,
    prompt_surface: dict,
    trajectory: dict,
    *,
    baseline_localized: dict | None = None,
) -> dict:
    recall_events = [
        event
        for event in _memory_events(events)
        if _is_memory_recall_event(event) and event.get("output", {}).get("success") is not False
    ]
    successful = [event for event in recall_events if _memory_result_count((event.get("output") or {}).get("data")) > 0]
    first = successful[0] if successful else None
    recall_surface = _recall_surface(successful)
    recalled_items = set(recall_surface["files"]) | set(recall_surface["identifiers"])
    prompt_items = set(prompt_surface["files"]) | set(prompt_surface["identifiers"])
    gold_identifiers = _gold_changed_identifiers(gold_patch)
    prompt_repetition = _safe_div(len(recalled_items & prompt_items), len(recalled_items))
    patch_relevance = _safe_div(len(set(recall_surface["identifiers"]) & gold_identifiers), len(recall_surface["identifiers"]))
    novel_patch_signals = (set(recall_surface["identifiers"]) & gold_identifiers) - set(prompt_surface["identifiers"])
    total_steps = int(trajectory.get("total_steps") or 0)
    recall_step = int(first.get("step")) if first else None
    action_follow = None
    post_gain = None
    baseline_expected_gain = None
    post_gain_delta = None
    if first:
        action_files = _files_between_steps(trajectory, recall_step + 1, recall_step + MEMORY_ACTION_WINDOW)
        if recall_surface["files"]:
            action_follow = _safe_div(len(set(recall_surface["files"]) & action_files), len(recall_surface["files"]))
        recall_by_step = trajectory.get("target_region_recall_by_step")
        if recall_by_step is None:
            localized = _localized_exploration_metrics(gold_patch, trajectory)
            recall_by_step = localized["target_region_recall_by_step"]
        before = _recall_at_step(recall_by_step, recall_step - 1)
        after = _recall_at_step(recall_by_step, recall_step + MEMORY_ACTION_WINDOW)
        post_gain = after - before
        if baseline_localized:
            baseline_recall_by_step = baseline_localized.get("target_region_recall_by_step") or []
            baseline_before = _recall_at_step(baseline_recall_by_step, recall_step - 1)
            baseline_after = _recall_at_step(baseline_recall_by_step, recall_step + MEMORY_ACTION_WINDOW)
            baseline_expected_gain = baseline_after - baseline_before
            post_gain_delta = post_gain - baseline_expected_gain
    return {
        "first_successful_recall_step": recall_step,
        "first_successful_recall_step_ratio": _round_or_none(_safe_div(recall_step, total_steps) if recall_step else None),
        "recall_parseable": bool(recalled_items),
        "recalled_files": len(recall_surface["files"]),
        "recalled_identifiers": len(recall_surface["identifiers"]),
        "recall_prompt_repetition_rate": _round_or_none(prompt_repetition),
        "recall_patch_identifier_relevance": _round_or_none(patch_relevance),
        "novel_patch_signal_count": len(novel_patch_signals),
        "memory_action_follow_rate_next5": _round_or_none(action_follow),
        "post_recall_target_region_gain_next5": _round_or_none(post_gain),
        "baseline_expected_target_region_gain_next5": _round_or_none(baseline_expected_gain),
        "post_recall_target_region_gain_delta_vs_baseline_next5": _round_or_none(post_gain_delta),
    }


def _repo_from_instance(instance_id: str) -> str:
    instance_id = instance_id.removeprefix("instance_")
    if "__" not in instance_id:
        return instance_id
    owner, rest = instance_id.split("__", 1)
    match = re.match(r"(.+?)(?:-[0-9a-f]{7,}.*|-v[0-9a-f]{7,}.*)?$", rest)
    return f"{owner}__{match.group(1) if match else rest}"


def _step_bucket(step_index) -> str:
    if step_index is None:
        return "unknown"
    step = int(step_index)
    if step <= 1:
        return "1"
    if step <= 3:
        return "2-3"
    if step <= 7:
        return "4-7"
    if step <= 15:
        return "8-15"
    return "16+"


def _prompt_surface(instance: dict) -> dict:
    text = "\n".join(
        str(instance.get(key) or "") for key in ("problem_statement", "requirements", "interface")
    )
    return {"files": sorted(_extract_files_from_text(text)), "identifiers": sorted(_extract_identifiers(text))}


def _recall_surface(events: list[dict]) -> dict:
    files: set[str] = set()
    identifiers: set[str] = set()
    for event in events:
        output = event.get("output") or {}
        text = _memory_output_text(output.get("data")) or output.get("raw") or ""
        files.update(_extract_files_from_text(str(text)))
        identifiers.update(_extract_identifiers(str(text)))
    return {"files": sorted(files), "identifiers": sorted(identifiers)}


def _memory_output_text(data: Any) -> str:
    if isinstance(data, dict):
        parts: list[str] = []
        for key in ("content", "answer", "text", "summary"):
            if data.get(key):
                parts.append(str(data[key]))
        for key in ("sessions", "results", "memories", "matches"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        parts.extend(str(item.get(k)) for k in ("content", "answer", "text", "summary", "path") if item.get(k))
                    else:
                        parts.append(str(item))
        return "\n".join(parts)
    if isinstance(data, list):
        return "\n".join(_memory_output_text(item) for item in data)
    return str(data or "")


def _extract_files_from_text(text: str) -> set[str]:
    return {_normalize_path(match.group(0).strip("`'\".,:;()[]{}")) for match in _PATH_RE.finditer(text or "")}


def _extract_identifiers(text: str) -> set[str]:
    scrubbed = _PATH_RE.sub(" ", text or "")
    identifiers = set()
    for match in _IDENT_RE.finditer(scrubbed):
        token = match.group(0)
        if len(token) < 3 or token.lower() in _STOPWORDS:
            continue
        identifiers.add(token)
    return identifiers


def _gold_changed_identifiers(gold_patch: str) -> set[str]:
    identifiers: set[str] = set()
    for hunk in _parse_patch_hunks(gold_patch):
        identifiers.update(_extract_identifiers("\n".join(hunk.get("changed_text") or [])))
    return identifiers


def _normalize_path(path: str) -> str:
    path = str(path or "").strip().strip("'\"`")
    if path.startswith("/testbed/"):
        path = path[len("/testbed/") :]
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("/"):
        path = path.lstrip("/")
    return path


def _target_regions(gold_patch: str, window: int) -> list[dict]:
    regions = []
    for index, hunk in enumerate(_eligible_hunks(_parse_patch_hunks(gold_patch))):
        start, end = _hunk_interval(hunk, window)
        regions.append({"index": index, "file": hunk["file"], "start": start, "end": end})
    return regions


def _regions_by_file(regions: list[dict]) -> dict[str, list[tuple[int, int]]]:
    by_file: dict[str, list[tuple[int, int]]] = {}
    for region in regions:
        by_file.setdefault(region["file"], []).append((int(region["start"]), int(region["end"])))
    return {file_path: _merge_intervals(intervals) for file_path, intervals in by_file.items()}


def _hit_target_region_indexes(
    lines_by_file: dict[str, list[tuple[int, int]]],
    target_regions: list[dict],
) -> set[int]:
    hits: set[int] = set()
    for region in target_regions:
        for interval in lines_by_file.get(region["file"], []):
            if _intervals_overlap(interval, (region["start"], region["end"])):
                hits.add(int(region["index"]))
                break
    return hits


def _files_between_steps(trajectory: dict, start: int, end: int) -> set[str]:
    files = set()
    for step, step_files in (trajectory.get("files_by_step") or {}).items():
        if start <= int(step) <= end:
            files.update(step_files)
    for step, by_file in (trajectory.get("lines_by_step") or {}).items():
        if start <= int(step) <= end:
            files.update(by_file)
    return files


def _recall_at_step(recall_by_step: list[float], step: int) -> float:
    if not recall_by_step or step <= 0:
        return 0.0
    index = min(step, len(recall_by_step)) - 1
    return float(recall_by_step[index])


def _union_lines(lines_by_step: dict[int, dict[str, list[tuple[int, int]]]]) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for by_file in lines_by_step.values():
        for file_path, intervals in by_file.items():
            result[file_path] = _merge_intervals(result.get(file_path, []) + intervals)
    return result


def _subtract_regions(
    lines_by_file: dict[str, list[tuple[int, int]]],
    regions_by_file: dict[str, list[tuple[int, int]]],
) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for file_path, intervals in lines_by_file.items():
        remaining = intervals
        for region in regions_by_file.get(file_path, []):
            remaining = _subtract_interval_list(remaining, region)
        if remaining:
            result[file_path] = remaining
    return result


def _subtract_interval_list(
    intervals: list[tuple[int, int]],
    remove: tuple[int, int],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in intervals:
        if not _intervals_overlap((start, end), remove):
            result.append((start, end))
            continue
        if start < remove[0]:
            result.append((start, remove[0] - 1))
        if end > remove[1]:
            result.append((remove[1] + 1, end))
    return _merge_intervals(result)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _line_total(lines_by_file: dict[str, list[tuple[int, int]]]) -> int:
    return sum(end - start + 1 for intervals in lines_by_file.values() for start, end in _merge_intervals(intervals))


def _line_intersection_total(
    a: dict[str, list[tuple[int, int]]],
    b: dict[str, list[tuple[int, int]]],
) -> int:
    total = 0
    for file_path in set(a) | set(b):
        total += _interval_intersection_length(a.get(file_path, []), b.get(file_path, []))
    return total


def _interval_intersection_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total = 0
    for left in _merge_intervals(a):
        for right in _merge_intervals(b):
            start = max(left[0], right[0])
            end = min(left[1], right[1])
            if start <= end:
                total += end - start + 1
    return total


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _f1(recall: float | None, precision: float | None) -> float | None:
    if recall is None or precision is None:
        return None
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _redundancy(unique_size: int, summed_size: int) -> float | None:
    if summed_size <= 0:
        return None
    return 1 - unique_size / summed_size


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _summary(
    instances: dict[str, InstanceMetrics],
    *,
    eval_results: dict[str, bool],
    baseline_results: dict[str, bool],
) -> dict:
    items = list(instances.values())
    labels = Counter(i.comparison_label for i in items)
    return {
        "total": len(items),
        "coverage": {
            "trajectories": len(items),
            "eval_results": sum(1 for i in items if i.passed is not None),
            "missing_eval_results": sum(1 for i in items if i.passed is None),
            "baseline_results": sum(1 for i in items if i.baseline_passed is not None),
            "missing_baseline_results": sum(1 for i in items if i.baseline_passed is None),
            "extra_eval_results": max(0, len(eval_results) - sum(1 for i in items if i.passed is not None)),
            "extra_baseline_results": max(
                0, len(baseline_results) - sum(1 for i in items if i.baseline_passed is not None)
            ),
        },
        "accuracy": _accuracy(i.passed for i in items),
        "baseline_accuracy": _accuracy(i.baseline_passed for i in items),
        "paired_vs_baseline": _paired_vs_baseline(instances),
        "comparison_labels": dict(sorted(labels.items())),
        "resources": _resource_totals(items),
        "resource_delta_vs_baseline": _resource_delta_summary(items),
        "tools": {
            "tool_calls": sum(i.process["tool_calls"] for i in items),
            "bash_calls": sum(i.process["bash_calls"] for i in items),
        },
        "patches": {
            "empty": sum(1 for i in items if i.outcome["empty_patch"]),
            "preds_traj_patch_mismatch": sum(1 for i in items if i.outcome["preds_traj_patch_mismatch"]),
        },
        "memory": _memory_summary(items),
        "patch_alignment": _metric_summary(items, "patch_alignment", [
            "edit_hunk_recall_w10",
            "edit_hunk_precision_w10",
            "edit_hunk_f1_w10",
        ]),
        "localized_exploration": {
            **_metric_summary(items, "localized_exploration", [
                "target_region_view_recall_w50",
                "target_region_view_precision_w50",
                "auc_target_region_recall_w50",
            ]),
            "target_region_found_rate": _round_or_none(
                _safe_div(
                    sum(1 for i in items if i.localized_exploration.get("target_region_found")),
                    sum(1 for i in items if i.localized_exploration.get("eligible")),
                )
            ),
            "target_region_never_found": sum(
                1
                for i in items
                if i.localized_exploration.get("eligible") and not i.localized_exploration.get("target_region_found")
            ),
        },
        "exploration_efficiency": _metric_summary(items, "exploration_efficiency", [
            "line_redundancy",
            "off_target_line_redundancy",
            "search_redundancy",
        ])
        | {
            "line_view_instances": sum(
                1 for i in items if i.localized_exploration.get("explicit_line_view_steps", 0) > 0
            ),
            "repeated_searches": sum(i.exploration_efficiency.get("repeated_searches", 0) for i in items),
        },
        "memory_influence": _metric_summary(items, "memory_influence", [
            "first_successful_recall_step_ratio",
            "recalled_files",
            "recalled_identifiers",
            "recall_prompt_repetition_rate",
            "recall_patch_identifier_relevance",
            "novel_patch_signal_count",
            "memory_action_follow_rate_next5",
            "post_recall_target_region_gain_next5",
            "baseline_expected_target_region_gain_next5",
            "post_recall_target_region_gain_delta_vs_baseline_next5",
        ])
        | {
            "novel_patch_signal_count": sum(i.memory_influence.get("novel_patch_signal_count", 0) for i in items),
            "parseable_recall_instances": sum(1 for i in items if i.memory_influence.get("recall_parseable")),
            "successful_recall_instances": sum(1 for i in items if i.memory.get("recall_successes", 0) > 0),
            "positive_post_recall_gain_delta_instances": sum(
                1
                for i in items
                if (i.memory_influence.get("post_recall_target_region_gain_delta_vs_baseline_next5") or 0) > 0
            ),
            "negative_post_recall_gain_delta_instances": sum(
                1
                for i in items
                if (i.memory_influence.get("post_recall_target_region_gain_delta_vs_baseline_next5") or 0) < 0
            ),
        },
        "repo_breakdown": _breakdown(instances, "repo"),
        "chain_breakdown": _breakdown(instances, "chain_id"),
        "step_breakdown": _breakdown(instances, "step_bucket"),
    }


def _accuracy(values) -> dict:
    values = list(values)
    evaluated = sum(v is not None for v in values)
    passed = sum(v is True for v in values)
    failed = sum(v is False for v in values)
    return {
        "evaluated": evaluated,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / evaluated, 4) if evaluated else None,
    }


def _resource_totals(items: list[InstanceMetrics]) -> dict:
    totals = {key: sum(i.process.get(key, 0) for i in items) for key in RESOURCE_KEYS}
    count = len(items) or 1
    totals["avg_total_tokens"] = round(totals["total_tokens"] / count, 2) if items else 0
    totals["avg_last_response_input_tokens"] = (
        round(sum(i.process.get("last_response_input_tokens", 0) for i in items) / count, 2) if items else 0
    )
    totals["avg_last_response_output_tokens"] = (
        round(sum(i.process.get("last_response_output_tokens", 0) for i in items) / count, 2) if items else 0
    )
    totals["avg_last_response_total_tokens"] = (
        round(sum(i.process.get("last_response_total_tokens", 0) for i in items) / count, 2) if items else 0
    )
    totals["avg_api_calls"] = round(totals["api_calls"] / count, 2) if items else 0
    totals["avg_instance_cost"] = round(totals["instance_cost"] / count, 6) if items else 0
    return totals


def _resource_delta(current: dict, baseline: dict) -> dict:
    return {key: current.get(key, 0) - baseline.get(key, 0) for key in RESOURCE_KEYS}


def _resource_delta_summary(items: list[InstanceMetrics]) -> dict:
    paired = [i for i in items if "delta_vs_baseline" in i.process]
    totals = {key: sum(i.process["delta_vs_baseline"].get(key, 0) for i in paired) for key in RESOURCE_KEYS}
    totals["paired_instances"] = len(paired)
    return totals


def _memory_summary(items: list[InstanceMetrics]) -> dict:
    tools = Counter()
    categories = Counter()
    totals = Counter()
    for item in items:
        memory = item.memory
        tools.update(memory["tools"])
        categories.update(memory["categories"])
        for key in (
            "tool_calls",
            "recall_attempts",
            "recall_successes",
            "recall_empty",
            "write_attempts",
            "write_successes",
            "tool_errors",
        ):
            totals[key] += memory.get(key, 0)
    return {
        "used_instances": sum(1 for i in items if i.memory["used"]),
        "successful_recall_instances": sum(1 for i in items if i.memory["recall_successes"] > 0),
        "tools": dict(sorted(tools.items())),
        "categories": dict(sorted(categories.items())),
        **{key: totals[key] for key in sorted(totals)},
    }


def _metric_summary(items: list[InstanceMetrics], attr: str, keys: list[str]) -> dict:
    summary: dict[str, float | int | None] = {}
    eligible_marker_seen = any("eligible" in getattr(item, attr) for item in items)
    for key in keys:
        values = [
            value
            for item in items
            if isinstance((value := getattr(item, attr).get(key)), int | float) and value is not None
        ]
        summary[f"avg_{key}"] = round(sum(values) / len(values), 4) if values else None
    if eligible_marker_seen:
        summary["eligible_instances"] = sum(1 for item in items if getattr(item, attr).get("eligible"))
    else:
        summary["eligible_instances"] = sum(
            1
            for item in items
            if any(isinstance(getattr(item, attr).get(key), int | float) for key in keys)
        )
    return summary


def _paired_vs_baseline(instances: dict[str, InstanceMetrics]) -> dict:
    with_baseline = [i for i in instances.values() if i.passed is not None and i.baseline_passed is not None]
    return {
        "both_passed": sum(1 for i in with_baseline if i.passed and i.baseline_passed),
        "current_only": sum(1 for i in with_baseline if i.passed and not i.baseline_passed),
        "baseline_only": sum(1 for i in with_baseline if not i.passed and i.baseline_passed),
        "both_failed": sum(1 for i in with_baseline if not i.passed and not i.baseline_passed),
    }


def _breakdown(instances: dict[str, InstanceMetrics], key: str) -> dict:
    groups: dict[str, list[InstanceMetrics]] = {}
    for metrics in instances.values():
        groups.setdefault(str(metrics.outcome.get(key) or "unknown"), []).append(metrics)
    return {
        group: {
            "total": len(items),
            "passed": sum(1 for i in items if i.passed),
            "pass_rate": round(sum(1 for i in items if i.passed) / len(items), 4) if items else None,
            "baseline_passed": sum(1 for i in items if i.baseline_passed),
            "memory_used": sum(1 for i in items if i.memory["used"]),
            "current_only": sum(1 for i in items if i.passed and i.baseline_passed is False),
            "baseline_only": sum(1 for i in items if i.passed is False and i.baseline_passed),
        }
        for group, items in sorted(groups.items())
    }
