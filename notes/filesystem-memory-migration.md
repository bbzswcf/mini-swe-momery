# 文件系统记忆（FileSystem Memory）迁移指南

把 mini-swe-agent 上已验证有效的 **filesystem memory** 迁移到更大、更复杂的 agent，用于其它 benchmark。
本文给出可移植的核心逻辑、必须对齐的接口契约，以及一份可直接执行的迁移检查清单。

> **本文自包含**：迁移所需的全部核心代码已**完整内联在文末附录**。下表的"原项目路径"仅用于溯源，**目标机器上不存在这些文件**——迁移时一切以附录代码为准，不要去引用这些路径。

| 角色 | 原项目路径（仅溯源，目标机无此文件） | 内联位置 |
| --- | --- | --- |
| 核心逻辑（几乎可整体搬运） | `memory/filesystem.py` | 附录 A |
| 复用的工具函数 `extract_message_text` | `memory/session_store.py` | 附录 B |
| 集成胶水：Manager / Agent / Runner / 容器挂载 | `memory/manager.py`、`agents/memory.py`、`agents/default.py`、`run/benchmarks/swebench.py` | 附录 C |
| 配置 overlay | `config/benchmarks/swebench_pro_filesystem.yaml` | 附录 D |
| 行为契约（迁移后据此校验） | `tests/memory/test_filesystem.py` | 附录 E |

---

## 1. 它是什么，和其它记忆有何不同

filesystem memory 是**链（chain）级别的、以普通文件 + bash 为载体**的经验记忆。一条 chain 是一串顺序执行、共享同一记忆目录的任务（instance）；后面的任务可以读前面任务沉淀下来的经验。

与同仓库另外两种记忆的关键区别（决定了它最好迁移）：

| | 读取方式 | 写入方式 | 是否注册工具 |
| --- | --- | --- | --- |
| `BuiltinMemory`（`MEMORY.md`） | 注入系统 prompt 的冻结快照 | 模型调用 `memory` 工具 | 是 |
| `session_search`（SQLite FTS） | 模型调用 `session_search` 工具 | session 结束自动索引 | 是 |
| **filesystem memory** | **模型用普通 `bash`（rg/grep/sed）读目录** | **任务结束由系统 rule 写入 + 一次 memory-only LLM 蒸馏** | **否** |

要点：

- **不暴露任何新工具**。唯一运行期工具仍是 `bash`。迁移时**无需改目标 agent 的工具 schema 或工具分发逻辑**。
- 读靠 prompt 注入一个**目录绝对路径**，模型自行用 shell 检索。
- 写发生在**任务结束之后**，分两部分：规则直接落盘的证据文件 + 一次模型蒸馏的经验摘要。任务进行中模型**不写**记忆。

---

## 2. 数据布局

记忆根为 `<home>/fs/chains/<chain_id>/`（`home` 缺省 `~/.mini-memory`，`chain_id` 经 `_safe_component` 清洗）：

```text
<home>/fs/chains/<chain_id>/
  README.md                      # 用法说明，含 MEMORY_CHAIN_DIR 精确路径
  INDEX.md                       # 所有 case 的表格索引（见下）
  repo.md                        # repo 级别可复用知识
  cases/<case_id>/
    task.md                      # 规则写入：任务原文（第一条 user 文本）
    trajectory.md                # 规则写入：完整轨迹，不截断
    patch.diff                   # 规则写入：最终 submission
    summary.md                   # 模型蒸馏摘要（剥离 Outcome），蒸馏失败时写结构化兜底
```

- `case_id` = `{step_index}-{session_id}`；无 `step_index` 时退化为 `{session_id}`。
- `INDEX.md` 表头固定：`| Step | Instance | Summary | Files / Symbols | Tests / Errors | Path |`，每个 case 一行，按 `Path` 列幂等 upsert。
- `repo.md` 骨架小节：`Test Commands` / `Repo Conventions` / `Repeated Patterns` / `Gotchas` / `Useful Files`；蒸馏返回**整份 repo.md**，宿主 sanitize 后**整文件覆盖**（合并去重交给蒸馏 LLM，不再有 `## Case Updates` 增量块）。
- 骨架文件（README/INDEX/repo）仅在缺失时创建（`_write_if_missing`），可跨任务累积。

---

## 3. 核心模块 `FileSystemMemory`：三个生命周期钩子

`FileSystemMemory` 是一个独立、几乎无外部耦合的类，迁移时基本可整体搬运。它只需要宿主在三个时机调用：

```python
@dataclass
class FileSystemMemoryConfig:
    home: Path | None = None      # 记忆根；None -> ~/.mini-memory
    enabled: bool = False
    chain_id: str = "default"

class FileSystemMemory:
    def initialize(self, session_id: str, **kwargs) -> None: ...
    def system_prompt_block(self) -> str: ...
    def on_session_end(self, messages: list[dict], *, model=None) -> dict: ...
```

1. **`initialize(session_id, chain_id=?, step_index=?)`** — 任务开始时调用。
   记录 `session_id`；可用 kwargs 在运行期覆盖 `chain_id`、设定 `step_index`；`enabled` 时建立目录骨架。

2. **`system_prompt_block()`** — 组装系统 prompt 时调用，返回注入文本（`enabled=False` 时返回 `""`）。
   内容是一个 `<filesystem_memory>` 块：给出 `chain_dir` 绝对路径、一行 `MEMORY_CHAIN_DIR=<quoted>` 复制命令、以及读取建议（把记忆当**提示而非真相**、与任务/代码冲突就忽略、先看 INDEX/repo/summary、trajectory 较长时先搜索再读目标片段、任务期间不要修改 `MEMORY_CHAIN_DIR` 下文件）。

3. **`on_session_end(messages, model=?)`** — 任务结束（无论成败）时调用。见第 4 节。

`disabled` 时三者均为 no-op 且不落任何文件（见 `test_disabled_filesystem_memory_is_noop`）。

---

## 4. 写路径（任务结束时 `on_session_end`）

分两段，**第二段需要一次 LLM 调用**：

**A. 规则直接落盘（无需模型，永远执行）**
- `task.md` ← 第一条 `role=="user"` 消息文本。
- `trajectory.md` ← `render_trajectory_markdown(messages)`，逐条 `## {idx}. {role}` 渲染，**完整不截断**。
- `patch.diff` ← 从后向前找到的 `extra.submission`。
- 最后统一 upsert 一行 INDEX（无 model 时各列为空），保证即使没有 model 也有索引条目。

**B. 模型蒸馏（仅当传入 `model`）**
- 用 `_summary_prompt(...)` 把 `task.md / trajectory.md / patch.diff / INDEX.md / repo.md` 全量拼进一个 prompt。
- 调用 `model.query_no_tools(...)`（不可用时回退 `model.query`），要求返回**单个 JSON**：
  ```json
  {"summary_md": "...", "index_row": {"summary": "...", "files_symbols": "...", "tests_errors": "..."}, "repo_md": "# Repo Notes\n..."}
  ```
- `summary_md` 写入 `cases/<case_id>/summary.md`；`index_row` upsert 进 `INDEX.md`；`repo_md` 是**整份 repo.md 全文**，经 sanitize（去围栏 / 剥 `## Outcome` 与 `## Case Updates` / 必须 `# Repo Notes` 开头 / 空则跳过）后**整文件覆盖**。
- `summary_md` 模板小节：`Task / Problem Signature / Investigation Path / Effective Change / Failed Attempts / Reusable Lessons`，每条经验需引用 trajectory/patch 证据。

**防泄漏（重要）**：摘要不应包含官方评测结论（resolved / pass / fail / 分数）。当前 mini 的蒸馏发生在判分**之前**，故 prompt 里**不再**写防泄漏指令，改由代码兜底——`_sanitize_summary_md`（summary）与 `_sanitize_repo_md`（repo）都用正则删除任何 `## Outcome` 段（见 `test_record_session_strips_outcome_section_from_model_summary`、`test_repo_md_is_overwritten_wholesale_and_sanitized`）。**⚠️ 一旦蒸馏时序变化能看到判分，必须把防泄漏指令补回 prompt。**

整个 B 段被 try/except 包裹：LLM 失败时记 `error` 并写一份**结构化兜底 `summary.md`** + 可读兜底 INDEX 行（保证 INDEX 不指向缺失文件），**绝不让记忆写入中断 agent 收尾**。蒸馏返回被散文/代码块包裹时，JSON 解析会回退正则抓取 `{...}`。

---

## 5. 读路径（任务进行中）

完全由模型用 `bash` 驱动，宿主不做检索：

1. `system_prompt_block()` 把 `chain_dir` 路径与策略注入系统 prompt。
2. 模型用 `rg`/`grep`/`find`/`sed` 检索该目录：先 `INDEX.md` → `repo.md` → `cases/*/summary.md`；`trajectory.md` 较长时先搜索、再读目标片段（注入块只要求"先搜再读"，不再硬性规定按行号 cat）。

**前置条件**：agent 执行 shell 的环境必须能以 prompt 中给出的**同一绝对路径**访问 `chain_dir`（见第 6.4 节容器挂载）。

---

## 6. 在 mini-swe-agent 里的集成（迁移时各对应什么）

### 6.1 `MemoryManager`（可选的聚合层）
manager 持有 `filesystem_memory`，并在 `initialize` / `system_prompt_block` / `on_session_end` 三处转发；config 里 `filesystem.home` 缺省继承 manager 的 `home`。
> 迁移可不照搬 manager。若目标 agent 已有自己的记忆聚合层，直接在其对应时机调用 `FileSystemMemory` 三钩子即可。

### 6.2 Agent 层（`MemoryAgent` extends `DefaultAgent`）
关键挂载点（迁移时在目标 agent 找对应位置）：
- **系统 prompt 注入**：`get_template_vars(memory_block=manager.system_prompt_block())`，系统模板里用 `{{ memory_block }}` 占位。
- **session 开始**：`run()` 起始 `manager.initialize(session_id, **kwargs)`（`kwargs` 透传 `chain_id`/`step_index`）。
- **session 结束**：`run()` 的 `finally` 里 `manager.on_session_end(self.messages, model=self.model)`。
- filesystem **不**参与 `execute_actions` 的工具分发（区别于 builtin/provider）。

### 6.3 Runner 的链式调度（`swebench.py`）
- `--chain-nodes <jsonl>`：每行一个节点 `{"chain_id": ..., "instance_id": ..., "step_index": ...}`。
- `load_chain_nodes` 按 `chain_id` 分组、按 `step_index` 升序排序；`order_instances_by_chains` 给每个 instance 打 `_chain_id` / `_step_index`。
- `chain_config`：每条链获得**独立** `memory.home = <memory_root>/<chain_id>` 与 `filesystem.chain_id = <chain_id>`。
- `process_chain`：同一条链内**顺序**执行（共享记忆）；链与链之间并行（`--chain-workers`）。
- `process_instance`：把 `session_id=instance_id`、`chain_id`、`step_index` 通过 `run_kwargs` 传入 `agent.run(task, **run_kwargs)`。

> 迁移核心：目标 agent 需要一个"链"调度器，保证**同链顺序、复用同一 `chain_id` 与独立 `home`**，并向每个任务提供 `session_id` 与 `step_index`。

### 6.4 容器路径挂载（关键，易漏）
任务在容器里跑时，模型 `bash` 必须能访问宿主上的 `chain_dir`。mini 在 `_mount_filesystem_memory_home` 里把宿主 `memory.home` 以**相同绝对路径**挂载进容器：

```text
docker run ... -v <host_home>:<host_home>:rw ...
```

> 迁移到其它执行后端（容器/远程/sandbox）时，**必须复刻这一点**：prompt 给的路径在执行 shell 的命名空间内可读可写，且 host 与容器路径一致。否则模型读不到记忆。

---

## 7. 配置

overlay（叠在基座 benchmark 配置之上）最小形态：

```yaml
agent:
  system_template: |
    ...你的系统提示...
    {{ memory_block }}
    ...
  memory:
    builtin_enabled: false      # 只跑 filesystem 时关掉其它记忆
    sessions_enabled: false
    provider: null
    filesystem:
      enabled: true
      chain_id: default         # runner 会按链覆盖
```

`MemoryManagerConfig` 相关键：`home`、`filesystem.{enabled, home, chain_id}`（`filesystem.home` 缺省继承 `home`）。

---

## 8. 消息结构契约（迁移必须对齐）

`on_session_end` 解析 `messages: list[dict]`，目标 agent 的消息需可映射到下列字段（否则证据文件会缺内容）：

| 用途 | 读取的字段 |
| --- | --- |
| `task.md` | 第一条 `role == "user"` 的文本 |
| `patch.diff` | 从后向前第一个 `extra["submission"]: str` |
| `trajectory.md` 每行 | `role`/`type` + `extract_message_text(msg)` |
| 工具/命令文本 | `extra["actions"] = [{"tool_name", "args": {"command": ...}}]` |
| Responses 风格 | `output[].{type, content, name, arguments}`、`tool_calls`、`tool_name` |

`extract_message_text`（来自 `session_store.py`）已兼容 chat 与 Responses 两种格式，并**刻意跳过** `extra.raw_output` 等超大旁路字段。迁移时优先复用它；若目标 agent 消息结构不同，改写这一个函数即可，`FileSystemMemory` 其余逻辑无需动。

---

## 9. 迁移检查清单

1. **搬运核心类**：复制**附录 A** 的 `filesystem.py`（`FileSystemMemory` + 全部 helper）。仅可能需要改：`extract_message_text` 的导入来源（见**附录 B**）、消息字段映射（第 8 节）。
2. **接 prompt 注入点**：在目标 agent 组装系统 prompt 处插入 `system_prompt_block()` 的返回值。
3. **接两个生命周期钩子**：任务开始调 `initialize(session_id, chain_id=, step_index=)`；任务结束（`finally`）调 `on_session_end(messages, model=)`。
4. **提供 memory-only LLM 接口**：实现/复用 `model.query_no_tools(messages)`（无则用 `query`），需能稳定返回**纯 JSON**，且**带重试**（蒸馏失败不应中断收尾）。
5. **实现链式调度**：同链顺序执行、复用 `chain_id`、每链独立 `home`（如 `<root>/<chain_id>`）；给每个任务传 `session_id` 与 `step_index`。
6. **保证路径可达**：执行 shell 的环境能以 prompt 中的绝对路径读写 `chain_dir`（容器需 `-v host:host:rw` 同路径挂载）。
7. **保留防泄漏过滤**：不把评测结论写进 `summary.md`（prompt 约束 + `_sanitize_summary_md`），尤其在有 ground-truth 的 benchmark。
8. **不要新增工具，且 prompt 同步**：filesystem 走 bash，无需注册/分发新工具；同时确保**最终生效**的 system / instance / format_error 模板只描述 bash + filesystem memory，不残留 `memory` / `session_search` 工具说明（基座配置里有，overlay 必须覆盖——详见**附录 D.1**）。
9. **断点恢复策略**：filesystem 写入幂等（按 case 路径 upsert，骨架仅缺失时建），可安全重跑同一 `chain_id` 而不污染历史。
10. **按附录 E 的行为契约校验**迁移后的实现。

---

## 10. 可移植性与依赖

- **纯可移植逻辑**：`filesystem.py` 全部（目录布局、prompt 文案、trajectory 渲染、INDEX/repo upsert、JSON 解析、防泄漏过滤）。仅依赖标准库（`json`/`re`/`shlex`/`pathlib`）+ `extract_message_text`。
- **需要按宿主适配的胶水**：prompt 注入点、生命周期钩子、链式调度、容器路径挂载、`model.query_no_tools` 与重试。
- **可丢弃**：`MemoryManager`、`BuiltinMemory`、`session_search`/SQLite、各 provider —— 与 filesystem 记忆相互独立，迁移 filesystem 时不必带上。

## 11. 验证

迁移后用**附录 E** 的等价用例确认：骨架与 prompt 生成、运行期选定 `chain_id`/`step_index`、规则证据文件 + 模型摘要、轨迹与摘要 prompt 均不截断、`repo.md` 全量重写 + sanitize、`query_no_tools` 优先、Outcome 段被剥离、蒸馏失败写结构化兜底 summary、JSON 解析放宽、无 model 时仍写证据与骨架 INDEX、disabled 为 no-op。

---

## 附录 A：`filesystem.py`（核心，可整体搬运）

迁移时**唯一**可能需要改的是顶部 import：把 `from minisweagent.memory.session_store import extract_message_text` 换成你项目中放置该函数的模块（函数本体见附录 B）。其余逻辑（目录布局、prompt 文案、轨迹渲染、INDEX/repo upsert、JSON 解析、防泄漏过滤）只依赖标准库，可原样使用。

````python
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
````

---

## 附录 B：`extract_message_text`（消息→纯文本，来自 `session_store.py`）

`filesystem.py` 唯一的项目内依赖。把消息扁平化为"LLM 实际看到的文本"，兼容 chat 与 Responses 两种格式，并**刻意跳过** `extra.raw_output` 等超大旁路字段。若目标 agent 的消息结构不同，**只需改这一个函数**，`FileSystemMemory` 其余逻辑无需动。

```python
import json


def extract_message_text(msg: dict) -> str:
    """Flatten mini/Responses/chat messages into the text the LLM actually saw.

    Off-band metadata such as ``extra.raw_output`` (the untruncated bash dump
    kept only for replay/debug; never sent to the model) is intentionally
    skipped: indexing it can blow past API content limits and pollutes recall
    with text the LLM never actually observed.
    """
    parts: list[str] = []
    _add_content(parts, msg.get("content"))
    _add_response_output(parts, msg.get("output"))
    if msg.get("type") == "function_call_output":
        _add_content(parts, msg.get("output"))
    if tool_name := msg.get("tool_name"):
        parts.append(f"tool:{tool_name}")
    if tool_calls := msg.get("tool_calls"):
        parts.append(_json_text({"tool_calls": tool_calls}))
    for action in (msg.get("extra") or {}).get("actions") or []:
        parts.append(_json_text({"tool": action.get("tool_name", "bash"), "args": action.get("args") or {}}))
    return "\n".join(p for p in parts if p).strip()


def _add_content(parts: list[str], content) -> None:
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                _add_content(parts, item.get("text") or item.get("output_text") or item.get("input_text"))
    elif content is not None:
        parts.append(str(content))


def _add_response_output(parts: list[str], output) -> None:
    if not isinstance(output, list):
        return
    for item in output:
        if not isinstance(item, dict):
            item = item.model_dump() if hasattr(item, "model_dump") else {}
        if item.get("type") == "function_call":
            parts.append(_json_text({"tool": item.get("name"), "args": item.get("arguments")}))
        elif item.get("type") == "function_call_output":
            _add_content(parts, item.get("output"))
        else:
            _add_content(parts, item.get("content"))


def _json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
```

---

## 附录 C：集成胶水（参考实现，按宿主 agent 改写）

下列代码展示 mini-swe-agent 如何把附录 A 接到 agent loop 与实验 runner。**不必照搬**——关键是在你自己的 agent 里复刻三件事：① 系统 prompt 注入 `system_prompt_block()`；② 任务开始/结束调 `initialize` / `on_session_end`；③ runner 做链式调度并保证记忆目录在执行 shell 内可达。

### C.1 在 agent 里挂三钩子

```python
# 任务开始：选定链、建骨架
filesystem_memory.initialize(session_id, chain_id=chain_id, step_index=step_index)

# 组装系统 prompt：把 block 注入（mini 用 jinja2 的 {{ memory_block }} 占位）
system_prompt = render(system_template, memory_block=filesystem_memory.system_prompt_block())

# 任务结束（务必放在 finally）：写证据 + 蒸馏摘要；model 需能返回纯 JSON，最好无 tools
try:
    run_agent_loop(...)
finally:
    filesystem_memory.on_session_end(messages, model=model)
```

mini 侧对应代码（`MemoryAgent` 继承自 `DefaultAgent`）：

```python
class MemoryAgent(DefaultAgent):
    def get_template_vars(self, **kwargs) -> dict:
        # 系统模板里写 {{ memory_block }}；DefaultAgent.run() 开头用 jinja2 渲染系统/任务模板
        return super().get_template_vars(memory_block=self.manager.system_prompt_block(), **kwargs)

    def run(self, task: str = "", session_id: str = "default", **kwargs) -> dict:
        self.manager.initialize(session_id, **kwargs)   # kwargs 透传 chain_id / step_index
        try:
            return super().run(task, **kwargs)
        finally:
            self.manager.on_session_end(self.messages, model=self.model)
```

`DefaultAgent` 的模板渲染时机（迁移到其它框架时找到等价位置即可）：

```python
def run(self, task: str = "", **kwargs) -> dict:
    self.extra_template_vars |= {"task": task, **kwargs}
    self.messages = []
    self.add_messages(
        self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
        self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
    )
    while True:
        self.step()                                   # 直到最后一条 role == "exit"
        if self.messages[-1].get("role") == "exit":
            break
    return self.messages[-1].get("extra", {})

def _render_template(self, template: str) -> str:
    from jinja2 import StrictUndefined, Template
    return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())
```

### C.2 可选的聚合层 `MemoryManager`（filesystem 相关分支）

若你的 agent 已有记忆聚合层可跳过本节，直接调附录 A 的三钩子。mini 的 manager 这样接线（`home` 缺省继承 manager 根目录）：

```python
# __init__：构建
if self.config.filesystem.enabled:
    fs_config = self.config.filesystem
    if fs_config.home is None or fs_config.home != self.config.home:
        fs_config = FileSystemMemoryConfig(
            home=fs_config.home or self.config.home,
            enabled=fs_config.enabled,
            chain_id=fs_config.chain_id,
        )
    self.filesystem_memory = FileSystemMemory(fs_config)
else:
    self.filesystem_memory = None

# initialize / system_prompt_block / on_session_end：转发
if self.filesystem_memory is not None:
    self.filesystem_memory.initialize(session_id, **kwargs)

if self.filesystem_memory is not None:
    parts.append(self.filesystem_memory.system_prompt_block())   # 拼进系统 prompt

if self.filesystem_memory is not None:
    try:
        self.filesystem_memory.on_session_end(messages, model=model)
    except Exception as exc:                                     # 记忆写入不得中断收尾
        logger.warning("filesystem memory on_session_end failed: %s", exc)

# from_config：解析 yaml 的 filesystem 段
if (filesystem := cfg.get("filesystem")) is not None:
    valid_fs_keys = {item.name for item in fields(FileSystemMemoryConfig)}
    fs_kwargs = {k: v for k, v in dict(filesystem).items() if k in valid_fs_keys}
    fs_kwargs["home"] = Path(str(fs_kwargs.get("home") or cfg.get("home") or Path.home() / ".mini-memory")).expanduser()
    mgr_kwargs["filesystem"] = FileSystemMemoryConfig(**fs_kwargs)
```

### C.3 Runner 的链式调度 + 容器挂载

链清单 JSONL 每行 `{"chain_id": ..., "instance_id": ..., "step_index": ...}`；同链顺序执行、共享独立 `home`，链间用线程池并行。

```python
def load_chain_nodes(path: Path) -> dict[str, list[dict]]:
    chains: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            node = json.loads(line)
            chains[node["chain_id"]].append(node)
    return {cid: sorted(nodes, key=lambda n: n["step_index"]) for cid, nodes in chains.items()}


def order_instances_by_chains(instances: list[dict], chain_nodes_path: Path) -> dict[str, list[dict]]:
    by_id = {inst["instance_id"]: inst for inst in instances}
    chains: dict[str, list[dict]] = {}
    for chain_id, nodes in load_chain_nodes(chain_nodes_path).items():
        chain_instances = []
        for node in nodes:
            if node["instance_id"] not in by_id:
                continue
            instance = dict(by_id[node["instance_id"]])
            instance["_chain_id"] = chain_id
            instance["_step_index"] = node["step_index"]
            chain_instances.append(instance)
        if chain_instances:
            chains[chain_id] = chain_instances
    return chains


def chain_config(config: dict, chain_id: str, memory_root: Path) -> dict:
    # 每条链一个独立 home：<memory_root>/<chain_id>
    memory_home = (memory_root / chain_id).expanduser().resolve()
    return recursive_merge(
        config,
        {"agent": {"memory": {"home": str(memory_home), "filesystem": {"chain_id": chain_id}}}},
    )


def _mount_filesystem_memory_home(config: dict, env_config: dict) -> None:
    # 关键：宿主 memory.home 以"相同绝对路径"挂进容器，模型 bash 才能按 prompt 给的路径读写
    memory_cfg = config.get("agent", {}).get("memory", {})
    if not memory_cfg.get("filesystem", {}).get("enabled") or not memory_cfg.get("home"):
        return
    host_home = Path(str(memory_cfg["home"])).expanduser().resolve()
    host_home.mkdir(parents=True, exist_ok=True)
    mount = f"{host_home}:{host_home}:rw"
    run_args = list(env_config.get("run_args") or ["--rm"])
    if mount not in run_args:
        run_args.extend(["-v", mount])
    env_config["run_args"] = run_args


def process_chain(chain_id, instances, output_dir, config, progress_manager, memory_root) -> None:
    config = chain_config(config, chain_id, memory_root)
    for instance in instances:           # 同一链顺序执行，共享 chain memory
        process_instance(instance, output_dir, config, progress_manager)


# process_instance 内：把 chain_id / step_index 透传给 agent.run
run_kwargs = {"session_id": instance_id}
if instance.get("_chain_id") is not None:
    run_kwargs["chain_id"] = instance["_chain_id"]
if instance.get("_step_index") is not None:
    run_kwargs["step_index"] = instance["_step_index"]
info = agent.run(task, **run_kwargs)
```

> `_mount_filesystem_memory_home` 在构建 docker 环境时调用。换其它执行后端（远程/sandbox）时，复刻"prompt 路径在执行 shell 内同路径可读写"这一条即可。本地直接跑（无容器）时记忆目录天然可达，无需挂载。

---

## 附录 D：配置 overlay（`swebench_pro_filesystem.yaml`）

`memory` 段是与记忆系统直接相关的部分，可整体迁移；`system_template` 必须含 `{{ memory_block }}` 占位；`instance_template` 是 benchmark 任务说明，迁到别的 benchmark 时整体替换。

```yaml
agent:
  # 系统模板：关键是包含 {{ memory_block }} 占位 —— system_prompt_block() 的注入点
  system_template: |
    You are a helpful assistant that can interact with a computer shell to solve programming tasks.

    {{ memory_block }}

    If filesystem memory is shown above, use bash to inspect it when earlier
    instances in this chain may help. Prefer INDEX.md, repo.md, and case
    summary.md files before reading targeted trajectory excerpts.

  # instance_template: |   # 任务说明（benchmark 特定）；迁到别的 benchmark 时整体替换。
  #   实际 overlay 的 instance_template 与本实验相关的三点：
  #   ① 只要求每轮至少一个 *bash* tool call（不提 memory / session_search 工具）；
  #   ② Recommended Workflow 含一句"若系统 prompt 出现 filesystem memory 策略，先简要查链记忆"；
  #   ③ 用固定 sentinel 提交：echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt，最终 patch 写入 extra.submission。

  memory:
    builtin_enabled: false      # 只跑 filesystem 时关掉内置 MEMORY.md 工具
    sessions_enabled: false     # 关掉 session_search（SQLite FTS）
    consolidation:
      on_session_end: false
      every_n_steps: 0
      max_actions: 3
      summary_max_chars: 4000
    provider: null              # 不用 Hindsight / Mem0
    filesystem:
      enabled: true
      chain_id: default         # runner 的 chain_config 会按链覆盖；也可手动指定

model:
  # 唯一运行期工具是 bash；错误重试提示也只提 bash，不要提任何记忆工具
  format_error_template: |
    Tool call error:

    <error>
    {{error}}
    </error>

    Every response needs to use at least one available tool. Use bash for shell
    work. If you have completed your assignment, consult the first message about
    how to submit your solution.
```

mini 侧的启动形态（供参考，链清单格式见附录 C.3 / 第 6.3 节）：

```bash
mini-extra swebench \
  -c <base_benchmark.yaml> \
  -c swebench_pro_filesystem.yaml \
  --chain-nodes <chain_nodes.jsonl> \
  --chain-workers <并行链数>
```

### D.1 prompt / 工具一致性核对（迁移与起跑前务必执行）

overlay 是**叠加**在基座 benchmark 配置之上的。基座（如 `swebench_pro.yaml`）的 `system_template` / `instance_template` / `model.format_error_template` 会向模型描述 `memory` 与 `session_search` 两个工具——但 filesystem 实验里这两个工具**并不注册**。因此 overlay **必须用自己的三份模板把基座版本完全覆盖**（`recursive_merge` 对字符串键是覆盖语义），否则模型会被告知有它根本调不到的工具，prompt 与实际能力不符 → 实验无效。**只改 `memory` 开关、不改这三处 prompt，是最常见的错误。**

源项目实测的合并结果（filesystem overlay 叠在 `swebench_pro.yaml` 上）应满足：

- `MemoryManager.tool_names` / `get_tool_schemas()` → **空**（无 `memory`、无 `session_search`、无 provider 工具；唯一运行期动作是基座提供的 `bash`）。
- 合并后 `system_template` / `format_error_template` → **只描述 bash + filesystem memory**，不含 `memory` 工具或 `session_search` 字样。
- 合并后 `instance_template` → overlay 版（不含 `session_search`；其中出现的 "memory" 仅指 *filesystem / chain memory* 检索，并非内置 `memory` 工具）。
- `system_prompt_block()`（`initialize` 之后）→ **只含一个 `<filesystem_memory>` 块**，无 builtin / provider 内容。

复现核对脚本（在目标项目按需改 import 与 spec 名）：

```python
import tempfile
from minisweagent.config import get_config_from_spec
from minisweagent.memory import MemoryManager
from minisweagent.utils.serialize import recursive_merge

cfg = recursive_merge(
    get_config_from_spec("swebench_pro.yaml"),
    get_config_from_spec("swebench_pro_filesystem.yaml"),
)
mem = dict(cfg["agent"]["memory"]); mem["home"] = tempfile.mkdtemp()
mgr = MemoryManager.from_config(mem)
assert mgr.tool_names == set(), mgr.tool_names            # 不暴露任何记忆工具
mgr.initialize("probe", chain_id="c", step_index=0)
assert mgr.system_prompt_block().startswith("<filesystem_memory>")
for bad in ("session_search", "MEMORY.md"):               # prompt 不得残留已禁用组件
    assert bad not in cfg["agent"]["system_template"]
    assert bad not in cfg["model"]["format_error_template"]
```

> 抽检 trajectory 同样有效：看 `messages[0]`（system）与首条 user 是否只描述 bash + filesystem memory，`extra.actions` 里出现的 tool 名是否都属于上面那个（空）集合 + `bash`。

---

## 附录 E：行为契约（迁移后逐条自测）

每条对应一个原始单测断言，迁移后应保持等价行为：

1. **初始化建骨架 + prompt**：`enabled` 后 `chain_dir` 下生成 `README.md`（每次 `initialize` 都覆盖刷新）/ `INDEX.md`（以 `# Chain Memory Index` 开头、不含 `## Repo-Level Notes`）/ `repo.md`（以 `# Repo Notes` 开头）；`system_prompt_block()` 含 `chain_dir` 绝对路径、`MEMORY_CHAIN_DIR=`、`Do not infer or shorten this path`、`Treat memory as advisory hints, not ground truth`、`If memory conflicts with the task or current code, ignore the memory.`、`Do not modify files under MEMORY_CHAIN_DIR during the task.`，且**不再含** `sed -n '<start>,<end>p'` / `Do not cat or read a whole trajectory.md` / `Never modify task.md`。
2. **运行期选链**：`initialize(session, chain_id="chain-a", step_index=4)` 后 `chain_dir` 指向 `chain-a`，`on_session_end` 落到 `cases/4-<session>/`。
3. **规则证据 + 模型摘要**：写 `task.md`（`# Task` 开头）、`trajectory.md`（含命令文本）、`patch.diff`（== submission）、`summary.md`；INDEX 行形如 `| 3 | <inst> | <summary> | <files> | <tests> | cases/3-<inst>/summary.md |`；摘要不含 `resolved` / `pass/fail`。
4. **不截断**：130k 字符的轨迹完整进 `trajectory.md`（无 `...[truncated]`），也完整进 summary prompt（mini 侧刻意保留不截断 / 不裁剪蒸馏输入）。
5. **repo.md 全量重写 + sanitize**：model 返回整份 `repo_md`，落盘前去掉可能的 markdown 代码块围栏、剥离 `## Outcome` 与 `## Case Updates` 段、要求以 `# Repo Notes` 开头（否则放弃本次更新）、空内容不更新；旧 `repo.md` 内容被**整体覆盖**，不再增量保留。
6. **`query_no_tools` 优先**：model 同时有 `query` 与 `query_no_tools` 时只调后者。
7. **防泄漏剥离**：model 返回的 `summary_md` 与 `repo_md` 中的 `## Outcome` 段都被删除，其它小节保留（prompt 层防泄漏指令已从蒸馏 prompt 移除，剥离改由代码兜底——**一旦蒸馏时序变化能看到判分，须把该 prompt 指令补回**）。
8. **无 model 仍可用**：`on_session_end(model=None)` 写证据文件 + 骨架 INDEX 行，`summary_written=False`。
9. **disabled no-op**：`enabled=False` 时三钩子无副作用，不创建 `fs/` 目录。
10. **蒸馏失败兜底**：蒸馏抛错或未返回可用 summary 时，写结构化兜底 `summary.md`（六小节、含失败原因与 patch 状态），并把对应 INDEX 行填成可读兜底值（`summary` 提示兜底、`files_symbols=task.md; trajectory.md; patch.diff`、`tests_errors=<失败原因截断>`），保证 INDEX 永不指向缺失文件；`summary_written` 仍为 `False`。
11. **JSON 解析放宽**：蒸馏响应被散文或 json 代码块包裹时仍能解析——直接 `json.loads` 失败则回退正则抓第一个 `{...}` 再解析。

最小自测骨架（把 import 换成你项目里的模块即可运行）：

```python
import json
from your_pkg.filesystem import FileSystemMemory, FileSystemMemoryConfig


class _JsonModel:
    def __init__(self, payload):
        self.payload, self.seen = payload, []

    def query(self, messages):
        self.seen.append(messages)
        return {"role": "assistant", "content": json.dumps(self.payload)}


def test_writes_evidence_and_summary(tmp_path):
    fs = FileSystemMemory(FileSystemMemoryConfig(home=tmp_path, enabled=True, chain_id="chain-a"))
    fs.initialize("repo__issue-1", step_index=3)
    model = _JsonModel(
        {
            "summary_md": "# repo__issue-1\n\n## Task\nFix parser crash.\n",
            "index_row": {"summary": "Fix parser crash", "files_symbols": "parser.py", "tests_errors": "pytest"},
            "repo_md": "# Repo Notes\n\n## Test Commands\n\n- run parser tests\n  Evidence: cases/3-repo__issue-1/summary.md\n",
        }
    )
    messages = [
        {"role": "user", "content": "Fix crash in parser.py; pytest test_parser.py fails."},
        {
            "role": "assistant",
            "content": "patch",
            "extra": {"actions": [{"tool_name": "bash", "args": {"command": "pytest test_parser.py"}}]},
        },
        {"role": "exit", "content": "Submitted", "extra": {"submission": "diff --git a/parser.py b/parser.py\n+fix\n"}},
    ]

    fs.on_session_end(messages, model=model)

    case = tmp_path / "fs" / "chains" / "chain-a" / "cases" / "3-repo__issue-1"
    assert (case / "task.md").read_text().startswith("# Task")
    assert "pytest test_parser.py" in (case / "trajectory.md").read_text()
    assert (case / "patch.diff").read_text() == "diff --git a/parser.py b/parser.py\n+fix\n"
    assert (case / "summary.md").read_text().startswith("# repo__issue-1")
    assert "cases/3-repo__issue-1/summary.md" in (case.parent.parent / "INDEX.md").read_text()
```
