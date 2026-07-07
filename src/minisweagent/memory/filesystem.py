"""Filesystem-backed chain memory for SWE-bench style runs.

This store is deliberately simple: Markdown files are the source of truth, and
the model is only allowed to write distilled notes after a task finishes. Raw
evidence files are rendered by rules.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minisweagent.memory.session_store import extract_message_text


@dataclass
class FileSystemMemoryConfig:
    home: Path | None = None
    enabled: bool = False
    chain_id: str = "default"


class FileSystemMemory:
    def __init__(self, config: FileSystemMemoryConfig | None = None) -> None:
        self.config = config or FileSystemMemoryConfig()
        self._session_id = ""
        self._step_index: int | None = None

    @property
    def chain_dir(self) -> Path:
        home = self.config.home or Path.home() / ".mini-memory"
        return home / "fs" / "chains" / _safe_component(self.config.chain_id)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        if kwargs.get("chain_id"):
            self.config.chain_id = str(kwargs["chain_id"])
        self._step_index = _coerce_step_index(kwargs.get("step_index"))
        if not self.config.enabled:
            return
        self._ensure_layout()

    def system_prompt_block(self) -> str:
        if not self.config.enabled:
            return ""
        quoted_chain_dir = shlex.quote(str(self.chain_dir))
        return (
            "<filesystem_memory>\n"
            f"You have filesystem memory for this experiment chain at:\n  {self.chain_dir}\n\n"
            "For shell commands, copy this exact assignment first:\n"
            f"  MEMORY_CHAIN_DIR={quoted_chain_dir}\n"
            "Do not infer or shorten this path; the useful files are under this exact directory.\n\n"
            "Use this chain-local memory when prior instances may help.\n"
            "- Treat memory as advisory hints, not ground truth; verify it against the current task and code.\n"
            "- If memory conflicts with the task or current code, ignore the memory.\n"
            "- Read repo.md for durable repo-level knowledge: test commands, conventions, repeated patterns, "
            "gotchas, and useful files.\n"
            "- Read INDEX.md to locate prior cases by instance, repo files, symbols, tests, errors, or issue keywords.\n"
            "- Open relevant cases/*/summary.md files after locating matching cases.\n"
            "- trajectory.md can be long; search it first and read only targeted excerpts when summaries are "
            "insufficient.\n\n"
            "Do not modify files under MEMORY_CHAIN_DIR during the task.\n"
            "</filesystem_memory>"
        )

    def on_session_end(self, messages: list[dict], *, model=None) -> dict:
        if not self.config.enabled:
            return {"enabled": False}
        self._ensure_layout()
        case_id = self._case_id()
        case_dir = self.chain_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        case_path = f"cases/{case_id}/summary.md"
        patch = _final_submission(messages)
        _write_text(case_dir / "task.md", f"# Task\n\n{_first_role_text(messages, 'user').strip()}\n")
        _write_text(case_dir / "trajectory.md", render_trajectory_markdown(messages))
        _write_text(case_dir / "patch.diff", patch)

        summary_written = False
        repo_updated = False
        model_error = ""
        index_row = {"summary": "", "files_symbols": "", "tests_errors": ""}

        if model is not None:
            try:
                payload = self._run_summary_model(model, case_dir)
                if summary := _sanitize_summary_md(str(payload.get("summary_md") or "")).strip():
                    _write_text(case_dir / "summary.md", summary + "\n")
                    summary_written = True
                if isinstance(row := payload.get("index_row"), dict):
                    index_row = {
                        "summary": str(row.get("summary") or ""),
                        "files_symbols": str(row.get("files_symbols") or ""),
                        "tests_errors": str(row.get("tests_errors") or ""),
                    }
                if repo_md := _sanitize_repo_md(str(payload.get("repo_md") or "")):
                    _write_text(self.chain_dir / "repo.md", repo_md)
                    repo_updated = True
            except Exception as exc:  # filesystem memory must not break shutdown
                model_error = str(exc)
            if not summary_written:  # always leave a structured summary so INDEX never dangles
                _write_text(case_dir / "summary.md", _fallback_summary_md(patch, model_error, case_path) + "\n")
                index_row = _fallback_index_row(model_error)

        self._upsert_index_row(step=self._step_label(), instance=self._session_id, path=case_path, **index_row)

        return {
            "enabled": True,
            "case_dir": str(case_dir),
            "summary_written": summary_written,
            "repo_updated": repo_updated,
            **({"error": model_error} if model_error else {}),
        }

    def _ensure_layout(self) -> None:
        (self.chain_dir / "cases").mkdir(parents=True, exist_ok=True)
        _write_text(self.chain_dir / "README.md", _readme_text(self.chain_dir))
        index_path = self.chain_dir / "INDEX.md"
        if index_path.exists():
            _migrate_index(index_path)
        else:
            _write_text(index_path, _index_skeleton())
        _write_if_missing(self.chain_dir / "repo.md", _repo_skeleton())

    def _case_id(self) -> str:
        session = _safe_component(self._session_id or "session")
        if self._step_index is None:
            return session
        return f"{self._step_index}-{session}"

    def _step_label(self) -> str:
        return "" if self._step_index is None else str(self._step_index)

    def _run_summary_model(self, model, case_dir: Path) -> dict:
        prompt = _summary_prompt(
            task=(case_dir / "task.md").read_text(),
            trajectory=(case_dir / "trajectory.md").read_text(),
            patch=(case_dir / "patch.diff").read_text(),
            index=(self.chain_dir / "INDEX.md").read_text(),
            repo=(self.chain_dir / "repo.md").read_text(),
            case_path=f"cases/{self._case_id()}/summary.md",
        )
        query = getattr(model, "query_no_tools", None) or model.query
        response = query([{"role": "user", "content": prompt}])
        return _parse_json_response(response)

    def _upsert_index_row(
        self,
        *,
        step: str,
        instance: str,
        summary: str,
        files_symbols: str,
        tests_errors: str,
        path: str,
    ) -> None:
        index_path = self.chain_dir / "INDEX.md"
        text = index_path.read_text() if index_path.exists() else _index_skeleton()
        row = (
            f"| {_cell(step)} | {_cell(instance)} | {_cell(summary)} | "
            f"{_cell(files_symbols)} | {_cell(tests_errors)} | {_cell(path)} |"
        )
        lines = text.splitlines()
        path_cell = f"| {_cell(path)} |"
        for i, line in enumerate(lines):
            if line.endswith(path_cell):
                lines[i] = row
                break
        else:
            lines.append(row)
        _write_text(index_path, "\n".join(lines).rstrip() + "\n")


def render_trajectory_markdown(messages: list[dict]) -> str:
    parts = ["# Trajectory\n"]
    for idx, msg in enumerate(messages):
        role = msg.get("role") or msg.get("type") or "message"
        text = extract_message_text(msg)
        if not text:
            continue
        parts.append(f"## {idx}. {role}\n\n{text.strip()}\n")
    return "\n".join(parts).rstrip() + "\n"


def _summary_prompt(*, task: str, trajectory: str, patch: str, index: str, repo: str, case_path: str) -> str:
    return (
        "You are updating filesystem memory for a sequence of coding tasks in one experiment chain.\n"
        "Return exactly one JSON object with keys: summary_md, index_row, repo_md.\n\n"
        "Rules:\n"
        "- summary_md must use sections: Task, Problem Signature, Investigation Path, Effective Change, "
        "Failed Attempts, Reusable Lessons.\n"
        "- index_row must contain summary, files_symbols, tests_errors.\n"
        "- repo_md must be the full updated contents of repo.md, starting with '# Repo Notes'.\n"
        "- Keep repo.md limited to durable repo-level knowledge in Test Commands, Repo Conventions, "
        "Repeated Patterns, Gotchas, and Useful Files. Avoid duplicating INDEX.md case summaries, file lists, "
        "or test logs.\n"
        "- Update repo.md only when this case provides durable evidence; preserve useful older notes, merge "
        "duplicates, and return repo_md unchanged when there is no durable repo-level lesson.\n"
        "- Do not create or preserve a Case Updates section.\n"
        "- Cite reusable lessons and repo.md bullets with evidence from summary.md, trajectory.md, or "
        f"patch.diff, using this case path: {case_path}\n\n"
        "<task.md>\n"
        f"{task}\n"
        "</task.md>\n\n"
        "<trajectory.md>\n"
        f"{trajectory}\n"
        "</trajectory.md>\n\n"
        "<patch.diff>\n"
        f"{patch}\n"
        "</patch.diff>\n\n"
        "<INDEX.md>\n"
        f"{index}\n"
        "</INDEX.md>\n\n"
        "<repo.md>\n"
        f"{repo}\n"
        "</repo.md>\n"
    )


def _parse_json_response(response: dict) -> dict:
    text = str(_response_text(response)).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        if not (match := re.search(r"\{.*\}", text, re.S)):
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("filesystem memory response must be a JSON object")
    return value


def _response_text(response: dict) -> str:
    content = response.get("content", "")
    if content:
        return _content_text(content)
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            text = _content_text(item.get("content", ""))
            if text:
                parts.append(text)
    return "\n".join(parts)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("output_text") or "")
            for item in content
            if isinstance(item, dict)
        )
    return str(content or "")


def _sanitize_summary_md(text: str) -> str:
    text = str(text or "")
    # The model is instructed not to include outcome/evaluation sections; strip
    # them defensively so downstream instances cannot see resolved/pass/fail
    # labels if the memory-only pass drifts.
    return re.sub(r"(?ims)^##+\s*Outcome\b.*?(?=^##+\s|\Z)", "", text).strip()


def _readme_text(chain_dir: Path) -> str:
    quoted_chain_dir = shlex.quote(str(chain_dir))
    return (
        "# Filesystem Memory\n\n"
        "Use this exact chain memory directory:\n\n"
        "```bash\n"
        f"MEMORY_CHAIN_DIR={quoted_chain_dir}\n"
        "cd \"$MEMORY_CHAIN_DIR\"\n"
        "```\n\n"
        "Use this directory as chain-local memory when prior instances may help.\n\n"
        "- Treat memory as advisory hints, not ground truth; verify it against the current task and code.\n"
        "- If memory conflicts with the task or current code, ignore the memory.\n"
        "- Read `repo.md` for durable repo-level knowledge: test commands, conventions, repeated patterns, "
        "gotchas, and useful files.\n"
        "- Read `INDEX.md` to locate prior cases by instance, repo files, symbols, tests, errors, or issue keywords.\n"
        "- Open relevant `cases/*/summary.md` files after locating matching cases.\n"
        "- `trajectory.md` can be very long; search it first and read only targeted excerpts when summaries are "
        "insufficient.\n"
        "- Do not modify memory files during the task.\n\n"
        "`task.md`, `trajectory.md`, and `patch.diff` are rule-owned evidence files.\n"
    )


def _index_skeleton() -> str:
    return (
        "# Chain Memory Index\n\n"
        "## Cases\n\n"
        "| Step | Instance | Summary | Files / Symbols | Tests / Errors | Path |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )


def _repo_skeleton() -> str:
    return (
        "# Repo Notes\n\n"
        "## Test Commands\n\n"
        "## Repo Conventions\n\n"
        "## Repeated Patterns\n\n"
        "## Gotchas\n\n"
        "## Useful Files\n"
    )


def _migrate_index(path: Path) -> None:
    """Strip the legacy default 'Repo-Level Notes' trailer from old INDEX.md files.

    Only the exact default boilerplate at end-of-file is removed; user-authored
    content under that heading is preserved.
    """
    text = path.read_text()
    migrated = re.sub(r"\n*##\s+Repo-Level Notes\s*\n+See `repo\.md`\.\s*\Z", "\n", text)
    if migrated != text:
        _write_text(path, migrated.rstrip() + "\n")


def _sanitize_repo_md(text: str) -> str:
    """Normalize a model-returned full repo.md: drop fences/Outcome/Case Updates; require the header."""
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(r"(?ims)^##+\s*Outcome\b.*?(?=^##+\s|\Z)", "", text)
    text = re.sub(r"(?ims)^##+\s*Case Updates\b.*?(?=^##+\s|\Z)", "", text).strip()
    if not text.startswith("# Repo Notes"):
        return ""
    return text.rstrip() + "\n"


def _fallback_summary_md(patch: str, reason: str, case_path: str) -> str:
    patch_state = (
        "A non-empty patch.diff was captured; inspect it with task.md and trajectory.md before reusing this case."
        if patch.strip()
        else "No patch content was captured for this case; inspect trajectory.md before reusing it."
    )
    return (
        "## Task\n"
        "Memory distillation failed for this case; see task.md for the full task text.\n\n"
        "## Problem Signature\n"
        "The memory-only distillation step did not produce a usable structured JSON summary. This file is an "
        "automatic fallback so INDEX.md never points at a missing summary.\n\n"
        "## Investigation Path\n"
        "- Raw task evidence is retained in task.md.\n"
        "- Full agent trajectory is retained in trajectory.md.\n"
        "- Captured code patch evidence is retained in patch.diff.\n\n"
        "## Effective Change\n"
        f"{patch_state}\n\n"
        "## Failed Attempts\n"
        f"- Memory distillation failed: `{_clip_reason(reason, 600)}`\n\n"
        "## Reusable Lessons\n"
        "- Treat this fallback as raw evidence only; verify task.md, trajectory.md, and patch.diff before "
        f"reuse. [evidence: {case_path}]"
    )


def _fallback_index_row(reason: str) -> dict:
    return {
        "summary": "Fallback summary: memory distillation failed; inspect raw evidence files before reuse.",
        "files_symbols": "task.md; trajectory.md; patch.diff",
        "tests_errors": _clip_reason(reason, 240),
    }


def _clip_reason(reason: str, max_chars: int) -> str:
    return re.sub(r"\s+", " ", str(reason or "")).replace("`", "'").strip()[:max_chars]


def _first_role_text(messages: list[dict], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return extract_message_text(msg)
    return ""


def _final_submission(messages: list[dict]) -> str:
    for msg in reversed(messages):
        extra = msg.get("extra") or {}
        submission = extra.get("submission")
        if isinstance(submission, str):
            return submission
    return ""


def _coerce_step_index(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_component(value: str) -> str:
    text = str(value or "default").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "default"


def _cell(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
