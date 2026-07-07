from __future__ import annotations

import json

from minisweagent.memory.filesystem import FileSystemMemory, FileSystemMemoryConfig


class _JsonModel:
    def __init__(self, payload: dict):
        self.payload = payload
        self.seen: list[list[dict]] = []

    def query(self, messages):
        self.seen.append(messages)
        return {"role": "assistant", "content": json.dumps(self.payload)}


class _NoToolsJsonModel:
    def __init__(self, payload: dict):
        self.payload = payload
        self.no_tools_seen: list[list[dict]] = []

    def query(self, messages):
        raise AssertionError("filesystem summary should use query_no_tools when available")

    def query_no_tools(self, messages):
        self.no_tools_seen.append(messages)
        return {
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": json.dumps(self.payload)}],
                }
            ],
        }


class _RaisingModel:
    def query_no_tools(self, messages):
        raise RuntimeError("boom: distillation unavailable")

    def query(self, messages):
        raise RuntimeError("boom: distillation unavailable")


class _RawTextModel:
    """Wraps the JSON in prose + ```json fences to exercise the relaxed parser."""

    def __init__(self, payload: dict):
        self.payload = payload

    def query_no_tools(self, messages):
        return {"role": "assistant", "content": f"Sure:\n```json\n{json.dumps(self.payload)}\n```\nDone."}


def _payload(**overrides) -> dict:
    payload = {
        "summary_md": "# repo__issue-1\n\n## Task\nFix parser crash.\n\n## Reusable Lessons\n- Lesson: run parser tests.\n  Evidence: trajectory.md\n",
        "index_row": {
            "summary": "Fix parser crash",
            "files_symbols": "parser.py",
            "tests_errors": "test_parser.py, ValueError",
        },
        "repo_md": "# Repo Notes\n\n## Test Commands\n\n- `python -m pytest test_parser.py`\n  Evidence: cases/3-repo__issue-1/summary.md\n",
    }
    payload.update(overrides)
    return payload


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Fix crash in parser.py when pytest test_parser.py fails."},
        {
            "role": "assistant",
            "content": "I will inspect the parser.",
            "extra": {"actions": [{"tool_name": "bash", "args": {"command": "sed -n '1,120p' parser.py"}}]},
        },
        {"role": "user", "content": "parser.py contents with ValueError"},
        {
            "role": "assistant",
            "content": "Patch parser.py and submit.",
            "extra": {
                "actions": [
                    {"tool_name": "bash", "args": {"command": "python -m pytest test_parser.py"}},
                ]
            },
        },
        {
            "role": "exit",
            "content": "Submitted",
            "extra": {"exit_status": "submitted", "submission": "diff --git a/parser.py b/parser.py\n+fix\n"},
        },
    ]


def test_filesystem_memory_initializes_chain_files_and_prompt(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-1")

    chain_dir = tmp_path / "fs" / "chains" / "chain-a"
    assert (chain_dir / "README.md").read_text()
    index_text = (chain_dir / "INDEX.md").read_text()
    assert index_text.startswith("# Chain Memory Index")
    assert "Repo-Level Notes" not in index_text
    assert (chain_dir / "repo.md").read_text().startswith("# Repo Notes")

    block = fs.system_prompt_block()
    assert str(chain_dir) in block
    assert "MEMORY_CHAIN_DIR=" in block
    assert "Do not infer or shorten this path" in block
    assert "Treat memory as advisory hints, not ground truth" in block
    assert "If memory conflicts with the task or current code, ignore the memory." in block
    assert "Do not modify files under MEMORY_CHAIN_DIR during the task." in block
    for removed in ("sed -n '<start>,<end>p'", "Do not cat or read a whole", "Never modify task.md"):
        assert removed not in block
    readme = (chain_dir / "README.md").read_text()
    assert "MEMORY_CHAIN_DIR=" in readme
    assert "Treat memory as advisory hints, not ground truth" in readme
    assert "rule-owned evidence files" in readme
    assert "sed -n" not in readme


def test_initialize_can_select_chain_id_at_runtime(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="default"))
    fs.initialize("repo__issue-1", chain_id="chain-a", step_index=4)

    assert fs.chain_dir == tmp_path / "fs" / "chains" / "chain-a"
    assert (fs.chain_dir / "README.md").exists()
    fs.on_session_end(_messages(), model=None)
    assert (fs.chain_dir / "cases" / "4-repo__issue-1" / "trajectory.md").exists()


def test_record_session_writes_rule_owned_case_files_and_model_owned_summary(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-1", step_index=3)
    model = _JsonModel(_payload())

    result = fs.on_session_end(_messages(), model=model)

    chain_dir = tmp_path / "fs" / "chains" / "chain-a"
    case_dir = chain_dir / "cases" / "3-repo__issue-1"
    assert result["case_dir"] == str(case_dir)
    assert result["summary_written"] is True and result["repo_updated"] is True
    assert (case_dir / "task.md").read_text().startswith("# Task")
    trajectory = (case_dir / "trajectory.md").read_text()
    assert "sed -n '1,120p' parser.py" in trajectory
    assert "python -m pytest test_parser.py" in trajectory
    assert (case_dir / "patch.diff").read_text() == "diff --git a/parser.py b/parser.py\n+fix\n"
    summary = (case_dir / "summary.md").read_text()
    assert summary.startswith("# repo__issue-1")
    assert "resolved" not in summary.lower() and "pass/fail" not in summary.lower()

    index = (chain_dir / "INDEX.md").read_text()
    assert "| 3 | repo__issue-1 | Fix parser crash | parser.py | test_parser.py, ValueError | cases/3-repo__issue-1/summary.md |" in index
    assert (chain_dir / "repo.md").read_text() == _payload()["repo_md"]  # wholesale overwrite


def test_summary_model_receives_complete_trajectory_without_truncation(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-long-summary", step_index=8)
    tail_marker = "TAIL_MARKER_IN_SUMMARY_PROMPT"
    model = _JsonModel(
        _payload(summary_md="# repo__issue-long-summary\n\n## Task\nFix long trace issue.\n", repo_md="")
    )

    fs.on_session_end(
        [
            {"role": "user", "content": "Fix long trace issue."},
            {"role": "assistant", "content": f"{'x' * 130_000}\n{tail_marker}"},
            {"role": "exit", "content": "Submitted", "extra": {"submission": "diff --git a/a b/a\n+fix\n"}},
        ],
        model=model,
    )

    prompt = model.seen[0][0]["content"]
    assert tail_marker in prompt
    assert "...[truncated]" not in prompt


def test_repo_md_is_overwritten_wholesale_and_sanitized(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-append", step_index=5)
    repo_path = tmp_path / "fs" / "chains" / "chain-a" / "repo.md"
    repo_path.write_text("# Repo Notes\n\n## Test Commands\n\n- stale command\n")
    model = _JsonModel(
        _payload(
            repo_md=(
                "```markdown\n# Repo Notes\n\n## Test Commands\n\n- `pytest -q`\n\n"
                "## Outcome\nResolved by passing tests.\n\n## Case Updates\n### old\n- legacy\n```"
            )
        )
    )

    fs.on_session_end(_messages(), model=model)

    repo = repo_path.read_text()
    assert repo.startswith("# Repo Notes")
    assert "- `pytest -q`" in repo
    assert "stale command" not in repo  # wholesale overwrite, not append
    assert "Outcome" not in repo and "Resolved by passing tests" not in repo  # leak guard
    assert "Case Updates" not in repo  # legacy mechanism stripped


def test_repo_md_rejected_when_header_missing(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-bad", step_index=6)
    repo_path = tmp_path / "fs" / "chains" / "chain-a" / "repo.md"
    skeleton = repo_path.read_text()

    result = fs.on_session_end(_messages(), model=_JsonModel(_payload(repo_md="## Test Commands\n\n- `pytest`\n")))

    assert result["repo_updated"] is False
    assert repo_path.read_text() == skeleton  # invalid repo_md ignored


def test_record_session_writes_complete_trajectory_without_truncation(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-long", step_index=9)
    long_middle = "x" * 130_000
    tail_marker = "TAIL_MARKER_SHOULD_REMAIN"

    fs.on_session_end(
        [
            {"role": "user", "content": "Fix long trace issue."},
            {"role": "assistant", "content": f"{long_middle}\n{tail_marker}"},
            {"role": "exit", "content": "Submitted", "extra": {"submission": "diff --git a/a b/a\n+fix\n"}},
        ],
        model=None,
    )

    trajectory = (
        tmp_path / "fs" / "chains" / "chain-a" / "cases" / "9-repo__issue-long" / "trajectory.md"
    ).read_text()
    assert tail_marker in trajectory
    assert "...[truncated]" not in trajectory
    assert len(trajectory) > 130_000


def test_summary_model_uses_no_tools_response_api_when_available(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-1", step_index=3)
    model = _NoToolsJsonModel(
        {
            "summary_md": "# repo__issue-1\n\n## Task\nFix parser crash.\n",
            "index_row": {
                "summary": "Fix parser crash",
                "files_symbols": "parser.py",
                "tests_errors": "test_parser.py",
            },
            "repo_md": "",
        }
    )

    fs.on_session_end(_messages(), model=model)

    case_dir = tmp_path / "fs" / "chains" / "chain-a" / "cases" / "3-repo__issue-1"
    assert model.no_tools_seen
    assert (case_dir / "summary.md").read_text().startswith("# repo__issue-1")
    assert "Fix parser crash" in (tmp_path / "fs" / "chains" / "chain-a" / "INDEX.md").read_text()


def test_record_session_strips_outcome_section_from_model_summary(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-1", step_index=1)
    model = _JsonModel(
        {
            "summary_md": (
                "# repo__issue-1\n\n"
                "## Task\nFix parser crash.\n\n"
                "## Outcome\nResolved by passing evaluation.\n\n"
                "## Reusable Lessons\n- Lesson: run focused tests.\n"
            ),
            "index_row": {"summary": "Fix parser crash", "files_symbols": "parser.py", "tests_errors": "pytest"},
            "repo_md": "",
        }
    )

    fs.on_session_end(_messages(), model=model)

    summary = (tmp_path / "fs" / "chains" / "chain-a" / "cases" / "1-repo__issue-1" / "summary.md").read_text()
    assert "## Outcome" not in summary
    assert "Resolved by passing evaluation" not in summary
    assert "## Reusable Lessons" in summary


def test_record_session_without_model_still_writes_evidence_and_skeleton_index(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-2")

    result = fs.on_session_end(_messages(), model=None)

    case_dir = tmp_path / "fs" / "chains" / "chain-a" / "cases" / "repo__issue-2"
    assert result["summary_written"] is False
    assert (case_dir / "task.md").exists()
    assert (case_dir / "trajectory.md").exists()
    assert (case_dir / "patch.diff").exists()
    index = (tmp_path / "fs" / "chains" / "chain-a" / "INDEX.md").read_text()
    assert "cases/repo__issue-2/summary.md" in index


def test_distillation_failure_writes_fallback_summary(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-fail", step_index=7)

    result = fs.on_session_end(_messages(), model=_RaisingModel())

    chain_dir = tmp_path / "fs" / "chains" / "chain-a"
    summary = (chain_dir / "cases" / "7-repo__issue-fail" / "summary.md").read_text()
    assert result["summary_written"] is False
    assert summary.startswith("## Task")
    assert "automatic fallback so INDEX.md never points at a missing summary" in summary
    assert "Memory distillation failed: `boom: distillation unavailable`" in summary
    index = (chain_dir / "INDEX.md").read_text()
    assert "Fallback summary: memory distillation failed" in index
    assert "cases/7-repo__issue-fail/summary.md" in index


def test_relaxed_json_parser_accepts_fenced_prose(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-raw", step_index=2)

    result = fs.on_session_end(_messages(), model=_RawTextModel(_payload()))

    chain_dir = tmp_path / "fs" / "chains" / "chain-a"
    assert result["summary_written"] is True
    assert (chain_dir / "cases" / "2-repo__issue-raw" / "summary.md").read_text().startswith("# repo__issue-1")
    assert "Fix parser crash" in (chain_dir / "INDEX.md").read_text()


def test_disabled_filesystem_memory_is_noop(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=False))
    fs.initialize("repo__issue-1")
    assert fs.system_prompt_block() == ""
    assert fs.on_session_end(_messages(), model=None) == {"enabled": False}
    assert not (tmp_path / "fs").exists()
