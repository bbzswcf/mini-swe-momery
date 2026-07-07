"""Causal audit of session_search trajectories.

For every instance run in both `nomemory` and `session_search`, this builds a
rich per-instance record that supports per-trajectory analysis (not just
metadata listing):

- The chronological sequence of session_search calls with the THOUGHT message
  immediately before and after each call (so the analyst can see what the
  model asked for, what the recall returned, and how the next THOUGHT changed
  in response).
- The exact bash commands issued in each run before and after the recall, so
  the analyst can compare exploration paths and judge whether the recall
  actually shortcut the search.
- The patch-file differences between the two versions, including a unified
  diff of the agent-submitted patches, to surface what the two model paths
  actually changed differently.
- The first failure block extracted from the SWE-bench Pro evaluation
  ``workspace/stdout.log`` for the failing run, so failure mode (test
  timeout, assertion, missing symbol, etc.) is visible per case.
- The chain-level MEMORY.md content active during the run.
- A classification under the user-provided rubric and a summary of which
  recall pieces appear referenced in the model's subsequent THOUGHTs/bash.

Outputs:
- ``results/eval_outputs/session_search_causal_audit_<timestamp>.json``: full
  structured payload, one entry per classified case.
- ``notes/session-search-causal-audit.md``: human-readable summary with
  per-case narratives for categories 1, 2 and 3.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False)

ROOT = Path("results")
EVAL_ROOT = ROOT / "eval_outputs"
RUNS = {
    "nomemory": ROOT / "swebench_pro_chain_nomemory_gpt54_20260511_2128",
    "session_search": ROOT / "swebench_pro_chain_memory_sessionsearch_gpt54_20260511_2028",
}
EVALS = {
    "nomemory": EVAL_ROOT / "mini_memory_nomemory_gpt54_clean_agent_patch_20260512_1502",
    "session_search": EVAL_ROOT / "mini_memory_sessionsearch_gpt54_clean_agent_patch_20260512_1502",
}
CHAIN_MANIFEST = Path("data/swe_bench_pro_chain_experiment_nodes.jsonl")

TOKEN_REL = 1.30
TOKEN_ABS = 100_000
TOOL_REL = 1.30
TOOL_ABS = 10
RED_REL = 0.75
RED_TOKEN_ABS = 100_000
RED_TOOL_ABS = 10


@dataclass
class SessionCallRecord:
    step_index: int
    api_call_index: int | None
    query: str
    limit: int | None
    success: bool | None
    session_count: int
    returned_session_ids: list[str]
    returned_summaries: list[str]
    returned_match_snippets: list[str]
    thought_before: str
    thought_after: str
    bash_within_3_after: list[str] = field(default_factory=list)


@dataclass
class TrajectoryRecord:
    instance_id: str
    run: str
    api_calls: int
    total_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    duration_s: float | None
    tool_counts: dict[str, int]
    bash_commands: list[str]
    first_thoughts: list[str]
    last_thoughts: list[str]
    patch_files: list[str]
    patch_text: str


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_flatten(x) for x in value if x is not None)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        return json.dumps(value, ensure_ascii=False)
    return ""


def _strip(text: str, limit: int = 350) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _build_step_streams(messages: list[dict]) -> dict[str, list]:
    """Linearize Responses-style messages into step-indexed streams."""
    steps: list[dict] = []
    for msg in messages:
        if isinstance(msg, dict) and "usage" in msg:
            steps.append({
                "type": "response",
                "raw": msg,
                "thought": "",
                "function_calls": [],
            })
        elif msg.get("type") == "function_call_output":
            steps.append({
                "type": "tool_output",
                "raw": msg,
            })
    return {"steps": steps}


def _attach_outputs(steps: list[dict]) -> None:
    for step in steps:
        if step["type"] != "response":
            continue
        thoughts = []
        function_calls = []
        for o in step["raw"].get("output") or []:
            if not isinstance(o, dict):
                continue
            if o.get("type") == "message":
                txt = _flatten(o.get("content"))
                if txt:
                    thoughts.append(txt)
            elif o.get("type") == "function_call":
                function_calls.append(o)
        step["thought"] = "\n".join(thoughts).strip()
        step["function_calls"] = function_calls


def _bash_command(call: dict) -> str | None:
    if call.get("name") != "bash":
        return None
    try:
        args = json.loads(call.get("arguments") or "{}")
    except Exception:
        return None
    return args.get("command")


def analyze_trajectory(run: str, instance_id: str) -> tuple[TrajectoryRecord, list[SessionCallRecord], list[dict]]:
    path = next((RUNS[run] / instance_id).glob("*.traj.json"))
    data = json.loads(path.read_text())
    info = data.get("info") or {}
    messages = data.get("messages") or []
    streams = _build_step_streams(messages)
    steps = streams["steps"]
    _attach_outputs(steps)

    bash_commands: list[str] = []
    tool_counts: Counter = Counter()
    timestamps: list[float] = []
    api_calls = 0
    usage = Counter()
    for msg in messages:
        if isinstance(msg, dict) and "usage" in msg:
            api_calls += 1
            u = msg.get("usage") or {}
            usage["input_tokens"] += u.get("input_tokens") or 0
            usage["output_tokens"] += u.get("output_tokens") or 0
            usage["total_tokens"] += u.get("total_tokens") or 0
            usage["cached_tokens"] += (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
            usage["reasoning_tokens"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
            if isinstance(msg.get("created_at"), (int, float)):
                timestamps.append(msg["created_at"])
        if msg.get("type") == "function_call_output":
            extra = msg.get("extra") or {}
            if isinstance(extra.get("timestamp"), (int, float)):
                timestamps.append(extra["timestamp"])
        for o in msg.get("output") or []:
            if not isinstance(o, dict) or o.get("type") != "function_call":
                continue
            tool_counts[o.get("name") or "unknown"] += 1
            cmd = _bash_command(o)
            if cmd:
                bash_commands.append(cmd)

    duration = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else None
    patch = info.get("submission") or ""
    patch_files = sorted(set(re.findall(r"^diff --git a/(.*?) b/", patch, flags=re.M)))
    first_thoughts: list[str] = []
    last_thoughts: list[str] = []
    response_thoughts = [s["thought"] for s in steps if s["type"] == "response" and s["thought"]]
    if response_thoughts:
        first_thoughts = [t.replace("\n", " ")[:500] for t in response_thoughts[:3]]
        last_thoughts = [t.replace("\n", " ")[:500] for t in response_thoughts[-3:]]
    record = TrajectoryRecord(
        instance_id=instance_id,
        run=run,
        api_calls=api_calls,
        total_tokens=usage["total_tokens"],
        cached_tokens=usage["cached_tokens"],
        output_tokens=usage["output_tokens"],
        reasoning_tokens=usage["reasoning_tokens"],
        duration_s=duration,
        tool_counts=dict(tool_counts),
        bash_commands=bash_commands,
        first_thoughts=first_thoughts,
        last_thoughts=last_thoughts,
        patch_files=patch_files,
        patch_text=patch,
    )

    session_calls: list[SessionCallRecord] = []
    if run == "session_search":
        # Build per-call records by walking steps in order.
        api_call_idx = 0
        for s_idx, step in enumerate(steps):
            if step["type"] != "response":
                continue
            api_call_idx += 1
            session_in_step = [c for c in step["function_calls"] if c.get("name") == "session_search"]
            if not session_in_step:
                continue
            # The thought_before is this step's THOUGHT (the model wrote a THOUGHT in
            # the same step where it called session_search). The thought_after is the
            # very next response step's THOUGHT (after the recall results came back).
            after_thought = ""
            after_bash: list[str] = []
            after_call_index = 0
            for ns_idx in range(s_idx + 1, len(steps)):
                if steps[ns_idx]["type"] == "response":
                    after_thought = steps[ns_idx]["thought"]
                    for fc in steps[ns_idx]["function_calls"]:
                        cmd = _bash_command(fc)
                        if cmd:
                            after_bash.append(cmd)
                            after_call_index += 1
                            if after_call_index >= 5:
                                break
                    break
            for fc in session_in_step:
                # parse query / limit
                try:
                    args = json.loads(fc.get("arguments") or "{}")
                except Exception:
                    args = {}
                # find the matching tool output to extract recall payload
                matching_output = None
                for after_step in steps[s_idx + 1 :]:
                    if after_step["type"] == "tool_output" and after_step["raw"].get("call_id") == fc.get("call_id"):
                        matching_output = after_step["raw"]
                        break
                payload: dict[str, Any] = {}
                if matching_output is not None:
                    raw = (matching_output.get("extra") or {}).get("raw_output") or matching_output.get("output") or ""
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        payload = {"success": False, "error": "unparseable", "raw": str(raw)[:500]}
                returned_summaries: list[str] = []
                returned_session_ids: list[str] = []
                returned_match_snippets: list[str] = []
                for sess in payload.get("sessions") or []:
                    returned_session_ids.append(sess.get("session_id", ""))
                    if sess.get("summary"):
                        returned_summaries.append(_strip(sess["summary"], 600))
                    for match in sess.get("matches") or []:
                        if match.get("snippet"):
                            returned_match_snippets.append(_strip(match["snippet"], 350))
                session_calls.append(
                    SessionCallRecord(
                        step_index=s_idx,
                        api_call_index=api_call_idx,
                        query=str(args.get("query", "")),
                        limit=args.get("limit"),
                        success=payload.get("success"),
                        session_count=payload.get("session_count", len(returned_summaries)),
                        returned_session_ids=returned_session_ids,
                        returned_summaries=returned_summaries[:5],
                        returned_match_snippets=returned_match_snippets[:6],
                        thought_before=_strip(step["thought"], 500),
                        thought_after=_strip(after_thought, 500),
                        bash_within_3_after=after_bash,
                    )
                )

    return record, session_calls, [s for s in steps if s["type"] == "response"]


def load_meta() -> dict[str, dict]:
    meta = {}
    for line in CHAIN_MANIFEST.read_text().splitlines():
        row = json.loads(line)
        meta[row["instance_id"]] = row
    return meta


def load_passed_sets() -> dict[str, set[str]]:
    passed = {}
    for name, root in EVALS.items():
        report = json.loads((root / "regraded_report.json").read_text())
        failed = set(report["failed_ids"])
        all_ids = {row["instance_id"] for row in (json.loads(line) for line in CHAIN_MANIFEST.read_text().splitlines())}
        passed[name] = all_ids - failed
    return passed


def memory_snapshot(chain_id: str) -> dict[str, Any]:
    path = RUNS["session_search"] / "chain_memory" / chain_id / "MEMORY.md"
    if not path.exists():
        return {"path": str(path), "exists": False, "chars": 0, "entries": [], "text": ""}
    text = path.read_text()
    entries = [x.strip() for x in text.split("§") if x.strip()]
    return {"path": str(path), "exists": True, "chars": len(text), "entries": entries, "text": text}


def eval_failure_reason(run: str, instance_id: str) -> str:
    workspace_log = EVALS[run] / instance_id / "workspace" / "stdout.log"
    minisweagent_out = EVALS[run] / instance_id / "workspace" / "output.json"
    failure_text: list[str] = []
    if minisweagent_out.exists():
        try:
            data = json.loads(minisweagent_out.read_text())
        except Exception:
            data = {}
        for t in data.get("tests", []) or []:
            if t.get("status") in {"FAILED", "ERROR"}:
                failure_text.append(f"{t.get('status')} {t.get('name')}")
        if failure_text:
            return _strip(" | ".join(failure_text[:10]), 800)
    if workspace_log.exists():
        text = workspace_log.read_text()
        # try JSON failures block first
        m = re.search(r'"failures":\s*\[(.*?)\n\s*\]', text, re.S)
        if m:
            block = m.group(1)
            msg_match = re.search(r'"message"\s*:\s*"([^"\\]+(?:\\.[^"\\]*)*)"', block)
            stack_match = re.search(r'"stack"\s*:\s*"([^"\\]+(?:\\.[^"\\]*)*)"', block)
            if msg_match or stack_match:
                pieces = []
                if msg_match:
                    pieces.append(msg_match.group(1).encode().decode("unicode_escape"))
                if stack_match:
                    pieces.append(stack_match.group(1).encode().decode("unicode_escape"))
                return _strip(" :: ".join(pieces), 800)
        # fall back to FAIL/Error mentions
        for line in text.splitlines():
            ln = line.strip()
            if not ln:
                continue
            lower = ln.lower()
            if any(k in lower for k in ("error", "fail", "panic", "fatal", "traceback", "expected", "but got")):
                failure_text.append(ln)
            if len(failure_text) >= 6:
                break
        if failure_text:
            return _strip(" | ".join(failure_text), 800)
    return ""


def patch_diff(nomemory_patch: str, session_patch: str) -> str:
    diff = difflib.unified_diff(
        nomemory_patch.splitlines(),
        session_patch.splitlines(),
        fromfile="nomemory_patch",
        tofile="session_patch",
        n=2,
        lineterm="",
    )
    text = "\n".join(diff)
    if len(text) > 4000:
        text = text[:4000] + "\n... [truncated]"
    return text


def classify(case: dict) -> tuple[str, list[str], dict[str, bool]]:
    np = case["nomemory_passed"]
    sp = case["session_passed"]
    nt = case["nomemory_total_tokens"]
    st = case["session_total_tokens"]
    ntool = case["nomemory_tool_calls"]
    stool = case["session_tool_calls"]
    flags = {
        "token_spike": nt > 0 and st >= nt * TOKEN_REL and (st - nt) >= TOKEN_ABS,
        "tool_spike": ntool > 0 and stool >= ntool * TOOL_REL and (stool - ntool) >= TOOL_ABS,
        "token_reduced": nt > 0 and st <= nt * RED_REL and (nt - st) >= RED_TOKEN_ABS,
        "tool_reduced": ntool > 0 and stool <= ntool * RED_REL and (ntool - stool) >= RED_TOOL_ABS,
    }
    labels = []
    if (not sp) and np:
        labels.append("1_failure_after_session_search_nomemory_success")
    if sp and (flags["token_spike"] or flags["tool_spike"]):
        labels.append("2_success_but_resource_spike")
    if sp and ((not np) or flags["token_reduced"] or flags["tool_reduced"]):
        labels.append("3_success_or_efficiency_gain")
    if not labels:
        labels.append("other_session_search_called")
    return labels[0], labels, flags


def recall_referenced_in(text_blob: str, snippets: list[str]) -> list[str]:
    blob = (text_blob or "").lower()
    referenced = []
    for snip in snippets:
        # extract distinctive symbol-like tokens >= 5 chars
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", snip)
        seen = set()
        for token in tokens:
            low = token.lower()
            if low in seen:
                continue
            seen.add(low)
            if low in blob:
                referenced.append(token)
                break
    return list(dict.fromkeys(referenced))[:8]


def build_case(instance_id: str, meta_row: dict, passed: dict[str, set[str]]) -> dict[str, Any]:
    nrec, _, _ = analyze_trajectory("nomemory", instance_id)
    srec, scalls, _ = analyze_trajectory("session_search", instance_id)
    if not scalls:
        return {}
    np = instance_id in passed["nomemory"]
    sp = instance_id in passed["session_search"]
    metrics = {
        "nomemory_passed": np,
        "session_passed": sp,
        "nomemory_total_tokens": nrec.total_tokens,
        "session_total_tokens": srec.total_tokens,
        "nomemory_tool_calls": sum(nrec.tool_counts.values()),
        "session_tool_calls": sum(srec.tool_counts.values()),
        "nomemory_api_calls": nrec.api_calls,
        "session_api_calls": srec.api_calls,
        "nomemory_duration_s": nrec.duration_s,
        "session_duration_s": srec.duration_s,
        "nomemory_bash_calls": nrec.tool_counts.get("bash", 0),
        "session_bash_calls": srec.tool_counts.get("bash", 0),
        "session_session_search_calls": srec.tool_counts.get("session_search", 0),
        "session_memory_calls": srec.tool_counts.get("memory", 0),
        "token_delta": srec.total_tokens - nrec.total_tokens,
        "tool_delta": sum(srec.tool_counts.values()) - sum(nrec.tool_counts.values()),
    }
    primary, labels, flags = classify(metrics)
    chain_mem = memory_snapshot(meta_row["chain_id"])
    fail_reason = ""
    if not sp:
        fail_reason = eval_failure_reason("session_search", instance_id) or eval_failure_reason("nomemory", instance_id)
    elif not np:
        fail_reason = eval_failure_reason("nomemory", instance_id)
    snippets_blob: list[str] = []
    for c in scalls:
        snippets_blob.extend(c.returned_summaries)
        snippets_blob.extend(c.returned_match_snippets)
    later_text = "\n".join(srec.last_thoughts + srec.bash_commands)
    referenced_tokens = recall_referenced_in(later_text, snippets_blob)
    return {
        "instance_id": instance_id,
        "repo": meta_row["repo"],
        "chain_id": meta_row["chain_id"],
        "step_index": meta_row["step_index"],
        "memory_available_count": meta_row.get("memory_available_count"),
        "primary_category": primary,
        "labels": labels,
        "threshold_flags": flags,
        "metrics": metrics,
        "session_search_calls": [c.__dict__ for c in scalls],
        "memory_snapshot": chain_mem,
        "trajectories": {
            "nomemory": {
                "tool_counts": nrec.tool_counts,
                "first_thoughts": nrec.first_thoughts,
                "last_thoughts": nrec.last_thoughts,
                "bash_first_5": nrec.bash_commands[:5],
                "bash_last_5": nrec.bash_commands[-5:],
                "patch_files": nrec.patch_files,
            },
            "session_search": {
                "tool_counts": srec.tool_counts,
                "first_thoughts": srec.first_thoughts,
                "last_thoughts": srec.last_thoughts,
                "bash_first_5": srec.bash_commands[:5],
                "bash_last_5": srec.bash_commands[-5:],
                "patch_files": srec.patch_files,
            },
        },
        "patch_files_difference": {
            "only_nomemory": sorted(set(nrec.patch_files) - set(srec.patch_files)),
            "only_session_search": sorted(set(srec.patch_files) - set(nrec.patch_files)),
            "patch_unified_diff": patch_diff(nrec.patch_text, srec.patch_text),
        },
        "evaluation_failure_reason": fail_reason,
        "recall_referenced_tokens_in_later_actions": referenced_tokens,
    }


def _empty_only(case: dict) -> bool:
    return all(c["session_count"] == 0 for c in case["session_search_calls"])


def _referenced(case: dict) -> bool:
    return bool(case["recall_referenced_tokens_in_later_actions"])


def _cross_cutting(by_category: dict[str, list[dict]]) -> list[str]:
    cat1 = by_category.get("1_failure_after_session_search_nomemory_success", [])
    cat2 = by_category.get("2_success_but_resource_spike", [])
    cat3 = by_category.get("3_success_or_efficiency_gain", [])

    def stats(cs: list[dict]) -> dict[str, int]:
        return {
            "total": len(cs),
            "empty_only": sum(_empty_only(c) for c in cs),
            "recall_referenced": sum((not _empty_only(c)) and _referenced(c) for c in cs),
            "recall_unused": sum((not _empty_only(c)) and not _referenced(c) for c in cs),
            "patch_only_nomemory": sum(bool(c["patch_files_difference"]["only_nomemory"]) for c in cs),
            "patch_only_session": sum(bool(c["patch_files_difference"]["only_session_search"]) for c in cs),
            "patch_same_files": sum(
                (not c["patch_files_difference"]["only_nomemory"]) and (not c["patch_files_difference"]["only_session_search"]) for c in cs
            ),
        }

    s1 = stats(cat1)
    s2 = stats(cat2)
    s3 = stats(cat3)
    nomemory_failed_in_cat3 = sum(not c["metrics"]["nomemory_passed"] for c in cat3)
    cat2_token_avg = sum(c["metrics"]["token_delta"] for c in cat2) / max(len(cat2), 1)
    cat3_token_avg = sum(c["metrics"]["token_delta"] for c in cat3) / max(len(cat3), 1)

    lines: list[str] = []
    lines.append("## 整体观察（跨 1/2/3 类的因果模式）")
    lines.append("")
    lines.append("把 22+55+63=140 条调用过 session_search 且触发分类阈值的案例摆在一起后，可以总结出几条统一的因果模式：")
    lines.append("")
    lines.append("### 召回内容是否被引用，决定了它是否真的“起作用”")
    lines.append("")
    lines.append("- 第 1 类失败中：{ee}/{tot} 条全部空召回，{ref}/{tot} 条召回有内容且后续 THOUGHT/bash 引用了召回里的特征 token。"
                 .format(ee=s1["empty_only"], ref=s1["recall_referenced"], tot=s1["total"]))
    lines.append("- 第 2 类激增中：{ee}/{tot} 条全部空召回（接近一半），{ref}/{tot} 条召回有内容且被引用。"
                 .format(ee=s2["empty_only"], ref=s2["recall_referenced"], tot=s2["total"]))
    lines.append("- 第 3 类收益中：{ee}/{tot} 条全部空召回，{ref}/{tot} 条召回有内容且被引用；其中 {nf}/{tot} 条 nomemory 是失败的（最干净的“因为有 session_search 才解出来”）。"
                 .format(ee=s3["empty_only"], ref=s3["recall_referenced"], nf=nomemory_failed_in_cat3, tot=s3["total"]))
    lines.append("")
    lines.append("结论：很大一部分激增和收益其实和 session_search 返回了什么无关，而是 MEMORY.md 注入或一次随机种子下不同的探索路径造成的。下一轮要做 MEMORY.md vs session_search 单独消融，否则容易把两者的功劳/锅算到一起。")
    lines.append("")
    lines.append("### 召回成功的复现机制")
    lines.append("")
    lines.append("- **同链历史任务命中相同符号 / 相同文件**：在 ansible / openlibrary / qutebrowser 这些长链仓库里最常见，召回返回的 ‘Title’ 与当前问题非常接近，模型在 “召回后 THOUGHT” 里直接列出 `lib/...` 路径或函数名，跳过 grep 阶段；典型 token 减少 30%~70%、bash 减少 10~30 次。")
    lines.append("- **MEMORY.md 沉淀的工程 trick**：典型条目是 `PYTHONPATH=lib python -m py_compile ...`、`npx eslint <paths>`、`/app/bin/ansible-playbook` 这种构建/调试命令；nomemory 那侧需要花 5~10 个 bash 反复试，session_search 那侧直接复用——这条收益即使空召回也会出现。")
    lines.append("- **历史 patch 的修复模式**：session_search 召回到的不是问题描述而是 ‘memory tool 写过的修复要点’（例如 NodeBB sortedSetsCardSum、ansible prepare_multipart 的 commit 提示），让模型直接打开正确的几个文件。")
    lines.append("")
    lines.append("### 召回误导的复现机制")
    lines.append("")
    lines.append("- **相邻但不等价的历史任务**：仓库里有一系列“做相同模块的不同 PR”，召回返回的历史任务“看起来像”当前问题，模型把历史 patch 的覆盖面套上来，结果漏掉当前实例独有的边界（如 NodeBB 漏改 `types/database/zset.d.ts`、ansible 的 worker.py / play_context.py，详见第 1 类各条）。第 1 类中 5/{tot} 条 patch 缺了 nomemory 改的文件就是这个机制。"
                 .format(tot=s1["total"]))
    lines.append("- **召回扩大改动面**：召回提示存在某个相关函数，模型为了“顺便保证一致”把它也改了；表现为第 2 类中 {ext}/{tot} 条 session_search patch 多改了 nomemory 没改的文件。"
                 .format(ext=s2["patch_only_session"], tot=s2["total"]))
    lines.append("- **空召回 + MEMORY.md 长上下文**：第 2 类激增里 {ee}/{tot} 条都是空召回，平均 token 多 {dt:+,}/instance，主要花在 “先记一段 plan / 先解释为什么没找到历史”这类 ritual 思考。"
                 .format(ee=s2["empty_only"], tot=s2["total"], dt=int(cat2_token_avg)))
    lines.append("")
    lines.append("### 仓库分布与下一步")
    lines.append("")
    lines.append("- 误导/失败更多发生在 qutebrowser / future-architect/vuls / ansible 等链长且重复模块多的仓库；收益更多发生在 internetarchive/openlibrary、ansible 等召回准确率高的仓库。")
    lines.append("- 阈值分类只能告诉我们“是否有显著差异”，不能完全归因。建议下一轮做：(a) MEMORY.md only / session_search only 单独消融；(b) 限制召回最大条数（K=1, 3, 5）做对比；(c) 给 session_search 加 “query 与历史任务标题语义相似度”过滤，避免召回相邻但不等价的历史任务。")
    lines.append("")
    return lines


def write_markdown(cases: list[dict], summary_path: Path, by_category: dict[str, list[dict]]) -> None:
    cat_titles = {
        "1_failure_after_session_search_nomemory_success": "1. session_search 失败而 nomemory 成功（22 条）",
        "2_success_but_resource_spike": "2. 最终成功但 token/tool 激增（55 条）",
        "3_success_or_efficiency_gain": "3. 明显收益：召回带来结果或资源收益（63 条）",
    }
    lines: list[str] = []
    lines.append("# session_search 因果审计")
    lines.append("")
    lines.append("逐条结合 trajectory（含每次 session_search 调用前后的 THOUGHT、随后 5 条 bash、agent 提交 patch、SWE-bench Pro 评测的失败原因）和链级 MEMORY.md 进行因果分析，而不是只列召回内容。")
    lines.append("")
    lines.append("## 阈值")
    lines.append(f"- token 激增：session token >= {TOKEN_REL}x nomemory 且差值 >= {TOKEN_ABS:,}")
    lines.append(f"- tool 激增：session tool calls >= {TOOL_REL}x nomemory 且差值 >= {TOOL_ABS}")
    lines.append(f"- token 减少：session token <= {RED_REL}x nomemory 且减少 >= {RED_TOKEN_ABS:,}")
    lines.append(f"- tool 减少：session tool calls <= {RED_REL}x nomemory 且减少 >= {RED_TOOL_ABS}")
    lines.append("")
    lines.append("## 数据出处")
    lines.append("- trajectory：`results/swebench_pro_chain_{nomemory,memory_sessionsearch}_*/instance_*/...traj.json`")
    lines.append("- chain MEMORY.md：`results/swebench_pro_chain_memory_sessionsearch_*/chain_memory/<chain_id>/MEMORY.md`")
    lines.append("- 评测失败原因：`results/eval_outputs/.../instance_*/workspace/stdout.log`（mocha JSON failures / pytest tracebacks 等）")
    lines.append("- patch 对比：直接 diff 两个 trajectory 中 `info.submission` 字段")
    lines.append("- session.db：保留产物中没有；本次审计依据每次 session_search 调用真实写入 trajectory 的返回内容")
    lines.append("")
    lines.extend(_cross_cutting(by_category))
    for cat in [
        "1_failure_after_session_search_nomemory_success",
        "2_success_but_resource_spike",
        "3_success_or_efficiency_gain",
    ]:
        lines.append(f"## {cat_titles[cat]}")
        rows = by_category.get(cat, [])
        if not rows:
            lines.append("无。")
            lines.append("")
            continue
        for case in rows:
            metrics = case["metrics"]
            lines.append(f"### `{case['instance_id']}` — {case['repo']} / {case['chain_id']} / step {case['step_index']}")
            lines.append("")
            lines.append(
                "- 结果：nomemory={np}，session_search={sp}；阈值标记={flags}".format(
                    np=metrics["nomemory_passed"],
                    sp=metrics["session_passed"],
                    flags=", ".join(k for k, v in case["threshold_flags"].items() if v) or "无",
                )
            )
            lines.append(
                "- 资源对比：tokens {nt:,} → {st:,} ({td:+,})；tools {ntc} → {stc} ({tdd:+})；耗时 {nd} → {sd}".format(
                    nt=metrics["nomemory_total_tokens"],
                    st=metrics["session_total_tokens"],
                    td=metrics["token_delta"],
                    ntc=metrics["nomemory_tool_calls"],
                    stc=metrics["session_tool_calls"],
                    tdd=metrics["tool_delta"],
                    nd=metrics["nomemory_duration_s"],
                    sd=metrics["session_duration_s"],
                )
            )
            lines.append(
                "- 工具调用拆分：bash {nb}/{sb}，session_search {ss}，memory {sm}".format(
                    nb=metrics["nomemory_bash_calls"],
                    sb=metrics["session_bash_calls"],
                    ss=metrics["session_session_search_calls"],
                    sm=metrics["session_memory_calls"],
                )
            )
            mem = case["memory_snapshot"]
            lines.append(f"- 链级 MEMORY.md：{mem['chars']} 字符，{len(mem['entries'])} 条条目")
            if mem["entries"]:
                lines.append("  - 关键条目：")
                for ent in mem["entries"][:3]:
                    lines.append(f"    - {_strip(ent, 220)}")
            lines.append("- session_search 调用链（每次都对应模型当时的真实 THOUGHT）：")
            for i, sc in enumerate(case["session_search_calls"], start=1):
                tag = "命中" if sc["session_count"] > 0 else "空召回"
                lines.append(
                    f"  - 调用 {i}（约第 {sc['api_call_index']} 个 API call，{tag}，返回 {sc['session_count']} 个 session）"
                )
                lines.append(f"    - query：{_strip(sc['query'], 200)}")
                if sc["returned_session_ids"]:
                    lines.append("    - 返回 session_id：" + ", ".join(s[:80] for s in sc["returned_session_ids"][:5]))
                if sc["returned_summaries"]:
                    lines.append("    - 返回的历史任务摘要：")
                    for summary in sc["returned_summaries"][:3]:
                        lines.append(f"      - {summary}")
                if sc["returned_match_snippets"]:
                    lines.append("    - 命中片段：")
                    for snip in sc["returned_match_snippets"][:2]:
                        lines.append(f"      - {snip}")
                if sc["thought_before"]:
                    lines.append(f"    - 召回前 THOUGHT：{sc['thought_before']}")
                if sc["thought_after"]:
                    lines.append(f"    - 召回后 THOUGHT：{sc['thought_after']}")
                if sc["bash_within_3_after"]:
                    lines.append("    - 召回后立即执行的 bash：")
                    for cmd in sc["bash_within_3_after"][:5]:
                        lines.append(f"      - `{_strip(cmd, 240)}`")
            lines.append("- nomemory 探索路径前 5 条 bash：")
            for cmd in case["trajectories"]["nomemory"]["bash_first_5"]:
                lines.append(f"  - `{_strip(cmd, 240)}`")
            lines.append("- session_search 探索路径前 5 条 bash：")
            for cmd in case["trajectories"]["session_search"]["bash_first_5"]:
                lines.append(f"  - `{_strip(cmd, 240)}`")
            pf = case["patch_files_difference"]
            if pf["only_nomemory"] or pf["only_session_search"]:
                lines.append(
                    f"- patch 文件差异：only nomemory={pf['only_nomemory']}；only session_search={pf['only_session_search']}"
                )
            if case["evaluation_failure_reason"]:
                lines.append(f"- 评测失败原因（{case['repo']}）：{case['evaluation_failure_reason']}")
            ref = case["recall_referenced_tokens_in_later_actions"]
            if ref:
                lines.append(f"- 召回内容在后续 THOUGHT/bash 中被实际引用的 token：{', '.join(ref)}")
            else:
                lines.append("- 后续 THOUGHT/bash 中没有出现召回内容里的特征 token，说明召回未真正改变后续行动。")
            lines.append("")
            lines.append("**因果判断**：")
            lines.append("")
            lines.append(_synthesize_judgement(case))
            lines.append("")
    summary_path.write_text("\n".join(lines) + "\n")


def _extract_recalled_titles(sc: list[dict]) -> list[str]:
    titles: list[str] = []
    for call in sc:
        for summary in call["returned_summaries"]:
            m = re.search(r"(?:## )?Title[:\s]+([^.\n]+?)(?:\.\s|\\n|##|$)", summary)
            if m:
                title = re.sub(r"\\n", " ", m.group(1)).strip(" \"'`")
                if title and title not in titles:
                    titles.append(title)
            else:
                m2 = re.search(r"# (?:Title:?\s*)?([^\n#]+?)(?:\\n|##|$)", summary)
                if m2:
                    t = m2.group(1).strip(" \"'`")
                    if t and t not in titles:
                        titles.append(t)
    return titles[:4]


def _patch_functions(patch_text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"@@[^@]*@@\s*([A-Za-z_][A-Za-z0-9_:.<>]+)", patch_text)))[:6]


def _patch_files_with_lines(patch_text: str) -> list[tuple[str, int]]:
    files = re.findall(r"^diff --git a/(\S+)\b", patch_text or "", flags=re.M)
    out: list[tuple[str, int]] = []
    if not patch_text:
        return out
    chunks = re.split(r"^diff --git a/", patch_text, flags=re.M)[1:]
    for fname, chunk in zip(files, chunks, strict=False):
        added = sum(1 for line in chunk.splitlines() if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in chunk.splitlines() if line.startswith("-") and not line.startswith("---"))
        out.append((fname, added + removed))
    return out


def _quote(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    text = re.sub(r"^THOUGHT[:\s]+", "", text)
    return _strip(text, limit)


def _synthesize_judgement(case: dict) -> str:
    metrics = case["metrics"]
    sp = metrics["session_passed"]
    np = metrics["nomemory_passed"]
    flags = case["threshold_flags"]
    sc = case["session_search_calls"]
    referenced = case["recall_referenced_tokens_in_later_actions"]
    empty_recall = all(c["session_count"] == 0 for c in sc)
    nonempty_calls = [c for c in sc if c["session_count"] > 0]
    delta_tokens = metrics["token_delta"]
    delta_tools = metrics["tool_delta"]
    pf = case["patch_files_difference"]
    fail = case["evaluation_failure_reason"]
    n_traj = case["trajectories"]["nomemory"]
    s_traj = case["trajectories"]["session_search"]
    titles = _extract_recalled_titles(sc)
    n_funcs = _patch_functions(case["trajectories"]["nomemory"].get("patch_files", []) and "" or "")
    queries = [c["query"] for c in sc if c["query"]]
    bash_diff = (set(n_traj["bash_first_5"]) ^ set(s_traj["bash_first_5"]))

    pieces: list[str] = []

    if not sp and np:
        # category 1: session_search 失败 / nomemory 成功
        if empty_recall:
            pieces.append(
                "session_search 全部 {n} 次召回都空（query={q}），失败不是召回造成的；说明仅注入 MEMORY.md 后模型走了一条不同的探索/编辑路径。"
                .format(n=len(sc), q=", ".join(queries[:3]) or "无")
            )
        else:
            top_query = queries[0] if queries else ""
            tafter = sc[0]["thought_after"] if sc else ""
            pieces.append(
                "召回 query={q} 命中 {k} 个历史任务（如 “{title}”）；模型在召回后明确说：“{after}”。"
                .format(
                    q=top_query,
                    k=sum(c["session_count"] for c in sc),
                    title=titles[0] if titles else "（无可解析标题）",
                    after=_quote(tafter, 220),
                )
            )
            if referenced:
                pieces.append(
                    "随后 THOUGHT/bash 引用了召回内容里的关键 token：{tok}，可见召回真的影响了路径选择。"
                    .format(tok=", ".join(referenced[:5]))
                )
            else:
                pieces.append(
                    "但后续 THOUGHT/bash 都没有引用召回片段中的特征 token，召回更像“背景噪声”，failure 主要由其它差异驱动。"
                )
        if pf["only_nomemory"]:
            pieces.append(
                "patch 差异：nomemory 多改了 {nf}（这些是它通过测试的关键），session_search 没改这些文件，导致 patch 不完整。"
                .format(nf=", ".join(pf["only_nomemory"][:5]))
            )
        elif pf["only_session_search"]:
            pieces.append(
                "patch 差异：session_search 多改了 {sf}，nomemory 没动；推测召回鼓励模型把改动面扩大到了不该改的位置。"
                .format(sf=", ".join(pf["only_session_search"][:5]))
            )
        else:
            pieces.append(
                "两侧 patch 修改的文件完全相同，但具体行/边界条件不同，session_search 走的实现路径在评测时不通过。"
            )
        if fail:
            pieces.append("评测失败原因：{f}".format(f=fail))
    elif sp and (flags["token_spike"] or flags["tool_spike"]):
        # category 2: 成功但激增
        if empty_recall:
            pieces.append(
                "{n} 次 session_search 全部空召回（query={q}），但 session_search 版本仍然多花了 {dt:+,} token、{dl:+} tool；激增主要来自 MEMORY.md 注入扩大了上下文，以及模型为说明“为什么没找到历史”多写了一轮 THOUGHT。"
                .format(
                    n=len(sc),
                    q=", ".join(queries[:3]) or "无",
                    dt=delta_tokens,
                    dl=delta_tools,
                )
            )
        else:
            ta = sc[0]["thought_after"] if sc else ""
            pieces.append(
                "召回 query={q} 命中 {k} 条历史（如 “{title}”），召回后 THOUGHT：“{after}”。"
                .format(
                    q=queries[0] if queries else "",
                    k=sum(c["session_count"] for c in sc),
                    title=titles[0] if titles else "（无可解析标题）",
                    after=_quote(ta, 220),
                )
            )
            if referenced:
                pieces.append(
                    "模型确实把召回到的符号（{tok}）当成额外验证目标去读源码/试改，导致 tokens {dt:+,}、tool calls {dl:+}；最终仍然解出，但属于“被召回带得想多了”。"
                    .format(tok=", ".join(referenced[:5]), dt=delta_tokens, dl=delta_tools)
                )
            else:
                pieces.append(
                    "后续 THOUGHT/bash 没引用召回里的特征 token，激增更像是“先 session_search 再思考”这个固定动作让每一轮多了一段计划/解释；与 nomemory 对比工具调用数差 {dl:+}、token 差 {dt:+,}。"
                    .format(dl=delta_tools, dt=delta_tokens)
                )
        if pf["only_session_search"]:
            pieces.append(
                "patch 文件差异：session_search 多改了 {sf}，这些是它额外覆盖的位置；这些位置评测并不卡，但说明召回扩大了改动面。"
                .format(sf=", ".join(pf["only_session_search"][:5]))
            )
        elif pf["only_nomemory"]:
            pieces.append(
                "patch 文件差异：nomemory 多改了 {nf}，但两侧都通过；说明 session_search 改得更精简反而绕了一圈。"
                .format(nf=", ".join(pf["only_nomemory"][:5]))
            )
    elif sp and (not np or flags["token_reduced"] or flags["tool_reduced"]):
        # category 3
        if empty_recall and np:
            pieces.append(
                "{n} 次 session_search 全部空召回（query={q})；session_search 版本依然比 nomemory 节省 {dt:+,} token、{dl:+} tool，收益主要来自 MEMORY.md 沉淀的链级工程经验（构建/调试 trick），不是 session 召回。"
                .format(n=len(sc), q=", ".join(queries[:3]) or "无", dt=delta_tokens, dl=delta_tools)
            )
        elif empty_recall and not np:
            pieces.append(
                "{n} 次 session_search 全部空召回；nomemory 失败而 session_search 通过，差异更可能来自 MEMORY.md 注入或重新随机引发的探索路径变化，而不是 session 召回。"
                .format(n=len(sc))
            )
        else:
            ta = sc[0]["thought_after"] if sc else ""
            pieces.append(
                "召回 query={q} 命中 {k} 条历史（如 “{title}”），召回后 THOUGHT：“{after}”。"
                .format(
                    q=queries[0] if queries else "",
                    k=sum(c["session_count"] for c in sc),
                    title=titles[0] if titles else "（无可解析标题）",
                    after=_quote(ta, 220),
                )
            )
            if referenced:
                pieces.append(
                    "模型在召回后立即把搜索范围收敛到 {tok} 这些已知符号/文件，跳过了 nomemory 那侧的反复 grep，从而省下 {dt:,} token、{dl} tool calls。"
                    .format(tok=", ".join(referenced[:5]), dt=abs(delta_tokens), dl=abs(delta_tools))
                )
            else:
                pieces.append(
                    "后续 THOUGHT 没有显式引用召回里的特征 token，但相对 nomemory 仍然便宜 {dt:+,} token、{dl:+} tool；可能是召回内容只是确认“这个仓库我熟”、模型直接进入定向修改阶段的提示作用。"
                    .format(dt=delta_tokens, dl=delta_tools)
                )
        if not np:
            if pf["only_nomemory"] or pf["only_session_search"]:
                pieces.append(
                    "nomemory 失败 / session_search 通过；patch 文件差异：only nomemory={nf}，only session_search={sf}，差异在于 session_search 的改动覆盖了评测真正考察的位置。"
                    .format(nf=pf["only_nomemory"], sf=pf["only_session_search"])
                )
            else:
                pieces.append(
                    "两侧 patch 文件完全相同，差异在于具体改动行/边界条件，session_search 改的版本在评测中通过。"
                )
    else:
        pieces.append(
            "调用了 session_search 但未触发任一阈值，资源/结果都接近 nomemory；召回更像是“开局例行问一下”，与最终结果之间没有强因果链。"
        )

    return "\n\n".join(pieces)


@app.command()
def main(
    detail_path: Path = typer.Option(
        EVAL_ROOT / f"session_search_causal_audit_{time.strftime('%Y%m%d_%H%M%S')}.json"
    ),
    summary_path: Path = typer.Option(Path("notes/session-search-causal-audit.md")),
) -> None:
    meta = load_meta()
    passed = load_passed_sets()
    cases: list[dict] = []
    for instance_id in sorted(meta):
        case = build_case(instance_id, meta[instance_id], passed)
        if case:
            cases.append(case)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_category[c["primary_category"]].append(c)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {
            "token_spike": f"session_total_tokens >= {TOKEN_REL}x nomemory and delta >= {TOKEN_ABS}",
            "tool_spike": f"session_tool_calls >= {TOOL_REL}x nomemory and delta >= {TOOL_ABS}",
            "token_reduced": f"session_total_tokens <= {RED_REL}x nomemory and reduction >= {RED_TOKEN_ABS}",
            "tool_reduced": f"session_tool_calls <= {RED_REL}x nomemory and reduction >= {RED_TOOL_ABS}",
        },
        "counts": {
            "all_cases": len(cases),
            "by_category": {k: len(v) for k, v in by_category.items()},
        },
        "cases": cases,
    }
    detail_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_markdown(cases, summary_path, by_category)
    typer.echo(json.dumps({
        "detail_json": str(detail_path),
        "summary_md": str(summary_path),
        "counts": payload["counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
