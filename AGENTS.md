# AGENTS

## 项目目标

构建面向代码任务（SWE-bench 类）的轻量记忆系统（mini-memory）：让 agent 解决 GitHub issue / repo-level 编码任务时，能跨 trial、跨 instance 复用学到的工程经验、踩过的坑、可行的改动模式。

非目标（决定了不迁移哪些东西）：

- **不做用户画像**：只服务一个自动化 agent，不迁移 hermes 的 `USER.md`。
- **不做 messaging / multi-user / 权限**：不接 gateway、Telegram、Discord。
- **不做 Skills（流程性知识）**：只做记忆（事实/经验性知识），不迁移 hermes 的 Skills 子系统。
- **不做多 profile / peer 建模**：单 agent 单实例足够。

## 实验口径

- memory 实验一律用链式子集 `data/swe_bench_pro_chain_experiment_nodes.jsonl`：链内按 `step_index` 顺序跑，链间独立、可按链并行。**不要**用全量 `data/swe_bench_pro.json`（731 条）做 memory 实验，它只作原始数据源。
- 调度：`mini-extra swebench --chain-nodes data/swe_bench_pro_chain_experiment_nodes.jsonl --chain-workers <链数>`；每条链自动用独立 `memory.home`。
- 默认基线测最小组合：`litellm_response` + `MemoryAgent` + 内置 `MEMORY.md` + 本地 SQLite FTS session recall，保持 `sessions_enabled: true`、`consolidation.on_session_end: false`。consolidation / Hindsight / Mem0 作为后续单独消融变量加入。
- 另有独立对照基线 **chain-window**（见下节）。

## 全量 no-memory baseline（731 条）

这是独立的全量无记忆基线，不属于上面的 memory 链式实验；因此这里明确使用 `data/swe_bench_pro.json` 的全部 731 条。完整流程必须包括 **inference → predictions 格式转换 → SWE-bench Pro harness 评测**，不能只跑 inference。

### 1. Inference

在本仓库根目录执行；模型 API key 从 `OPENAI_API_KEY` 环境变量读取，不写入仓库：

```bash
cd ~/code/mini-memory
export RUN="$PWD/results/swebench_pro_full_nomemory_gpt54"
export EVAL_REPO="$HOME/code/SWE-bench_Pro-os"

uv run --frozen mini-extra swebench \
  -c src/minisweagent/config/benchmarks/swebench_pro_nomemory.yaml \
  --subset data/swe_bench_pro.json \
  --workers 8 \
  -o "$RUN"
```

全量结束后先确认 `preds.json` 恰好包含 731 条；不足 731 条不能作为全量基线直接报分：

```bash
uv run --frozen python - <<'PY'
import json
import os
from pathlib import Path

preds = json.loads((Path(os.environ["RUN"]) / "preds.json").read_text())
print("predictions:", len(preds))
assert len(preds) == 731, f"Expected 731 predictions, got {len(preds)}"
PY
```

### 2. 转换为 SWE-bench Pro patch 格式

mini-swe-agent 的 `preds.json` 是以 `instance_id` 为 key 的字典，不能原样交给官方 Pro harness。用仓库内脚本转成 `[{"instance_id", "patch", "prefix"}, ...]`：

```bash
uv run --frozen python scripts/pro_preds_to_eval.py \
  --input "$RUN/preds.json" \
  --output "$RUN/patches.json" \
  --prefix nomemory-gpt54
```

当前 `SWE-bench_Pro-os/swe_bench_pro_eval.py` 接受 CSV 或 JSONL raw sample，但仓库不自带 README 示例中的 `swe_bench_pro_full.csv`。从本仓库已跟踪的 731 条 JSON 生成评测 JSONL：

```bash
uv run --frozen python - <<'PY'
import json
import os
from pathlib import Path

rows = json.loads(Path("data/swe_bench_pro.json").read_text())
output = Path(os.environ["RUN"]) / "swe_bench_pro_eval_input.jsonl"
with output.open("w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
print(f"Wrote {len(rows)} instances to {output}")
PY
```

### 3. 安装并固定 SWE-bench Pro evaluator

评测使用 `bbzswcf/SWE-bench_Pro-os`（当前核验 commit：`fdf26d5646055b8bfdf58f4f5c8a63c8fb796d18`）。新机器首次准备：

```bash
git clone https://github.com/bbzswcf/SWE-bench_Pro-os.git "$EVAL_REPO"
git -C "$EVAL_REPO" checkout fdf26d5646055b8bfdf58f4f5c8a63c8fb796d18
cd "$EVAL_REPO"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

若目录已存在，则不要重复 clone；只需核对 commit 并确保 `.venv` 依赖已安装。731 条数据需要的 `run_scripts/`、base Dockerfile 与 instance Dockerfile 在上述 evaluator commit 中均已核对齐全。

### 4. 本地 Docker 正式评测

必须从 `SWE-bench_Pro-os` 根目录启动，因为 evaluator 会通过相对路径读取 `dockerfiles/`：

```bash
cd "$EVAL_REPO"
.venv/bin/python swe_bench_pro_eval.py \
  --raw_sample_path "$RUN/swe_bench_pro_eval_input.jsonl" \
  --patch_path "$RUN/patches.json" \
  --output_dir "$RUN/eval" \
  --scripts_dir "$EVAL_REPO/run_scripts" \
  --num_workers 16 \
  --dockerhub_username jefzda \
  --use_local_docker
```

- 最终逐题 pass/fail 结果在 `$RUN/eval/eval_results.json`，逐题日志在 `$RUN/eval/<instance_id>/`；总分也会由 evaluator 打印到终端。
- 评测默认断点续跑：已有 `<prefix>_output.json` 的 instance 会跳过；仅在需要强制重评时加 `--redo`。
- inference 的 `--workers` 与 evaluation 的 `--num_workers` 是两套独立并发，按机器 CPU、内存、磁盘和 Docker 承载能力分别调整。
- 不要依赖本机曾经存在但未跟踪的 `SWE-bench_Pro-os/eval_swebench_pro` 包装器；标准可迁移流程是显式调用本仓库的 `scripts/pro_preds_to_eval.py` 和 evaluator 的 `swe_bench_pro_eval.py`。

### 新实验启动前：核对 prompt 与工具

消融 overlay（如 `swebench_pro_hindsight_only.yaml`）叠在 `swebench_pro.yaml` 上，**只改 `agent.memory` 开关不够**：若不覆盖 `system_template` / `instance_template` / `model.format_error_template`，模型仍按基座 prompt 里的 `memory` / `MEMORY.md` / `session_search` 行事，与实际暴露的工具不一致，实验作废。

每次新开跑或改 overlay 后，先用一条 traj 做 spot check：

1. **开关**（`agent.memory`）：`builtin_enabled` / `sessions_enabled` / `consolidation` / `provider` 与设计一致。
2. **合并后 prompt**（overlay + `swebench_pro.yaml`）：`system_template`（含 `{{ memory_block }}` 注入内容）、`instance_template`（工具名、CRITICAL 里「至少一个 tool call」列举）、`model.format_error_template`（重试提示里的工具列表）只描述本实验**实际启用**的能力。对照 `swebench_pro_nomemory.yaml`、`swebench_pro_hindsight_only.yaml`。
3. **实际注册的工具**（`MemoryManager.get_tool_schemas()` / `tool_names`）与 prompt 一一对应。例如纯 Hindsight 应为 `hindsight_retain` / `hindsight_recall` / `hindsight_reflect`，无 `memory`、无 `session_search`。
4. **Provider 工具 schema 文案**：`mirror_builtin_writes: false` 或 `builtin_enabled: false` 时，description 不得再提内置 `memory` 工具。
5. **Trajectory 抽检**：system 与首条 user 无已禁用组件的残留说明；`extra.actions` 里的 tool 名都属于第 3 步集合。

自检脚本（仓库根目录，替换 overlay 文件名）：

```bash
uv run python -c "
from minisweagent.config import get_config_from_spec
from minisweagent.memory import MemoryManager
from minisweagent.utils.serialize import recursive_merge
cfg = recursive_merge(
    get_config_from_spec('swebench_pro.yaml'),
    get_config_from_spec('swebench_pro_hindsight_only.yaml'),
)
m = cfg['agent']['memory']
print('tool_names:', MemoryManager.from_config(m).tool_names)
for k in ('system_template', 'instance_template', 'format_error_template'):
    text = cfg['agent'].get(k) or cfg.get('model', {}).get(k, '')
    print(f'--- {k} ---'); print(text[:600])
"
```

或直接读 `results/.../instance_*/*.traj.json` 的 `messages[0]`（system）与首条 user，对照 `extra.actions` 的 tool 名。

## Chain-window baseline（独立对照）

整条链共享一个 LLM context，不走 memory tool / `MEMORY.md`，靠阈值触发的对话压缩跨题复用上下文。走独立 agent + runner，与其他 memory 实验互不影响、可并行。

- 入口：`mini-extra swebench-chain-window -c src/minisweagent/config/benchmarks/swebench_pro_chain_window.yaml --chain-nodes data/swe_bench_pro_chain_experiment_nodes.jsonl --chain-workers <链数>`
- Agent：`ChainWindowAgent`（`src/minisweagent/agents/chain_window.py`，`DefaultAgent` 子类）——整条链共用 `self.messages`，首题装系统 prompt + 题目、后续题只追加 user message；`step_limit` / `cost_limit` 按每题计，`n_calls` / `cost` 累计全链。`input_tokens` 超过 `model_window * threshold` 时把"已完成的题"压成一条 `<compressed_history>`（正在做的题不动，旧 summary 被新的替换、不叠加），压缩走 `model.query_no_tools`（同主模型、无 tools、自带 retry）。具体阈值/字段见配置与 agent docstring。
- 配置 `swebench_pro_chain_window.yaml`：`compression.model_window`（默认 272000）/ `threshold`（0.8）/ 输出预算（`max_output_tokens`、`char_budget`）/ `trace_max_chars`；prompt 与 `swebench_pro_nomemory.yaml` 对齐，保证 baseline 干净。
- 断点恢复（复用同一 `-o` 目录）：跑完的链跳过；跑一半的链无法半恢复（题 N 依赖前面累计/压缩出的上下文），故整条清掉 preds + 各题 traj 重跑，日志开头打印 resume 统计。

## 目录约定

- `src/minisweagent/`：本项目源码，新增代码都写这里。
- `memory_repo/`：**只读参考仓库**目录。每个子目录是一个外部项目的完整 clone（保留各自 `.git`，非 submodule，未纳入本仓库 git 跟踪），仅供阅读对照，不要修改、不要 import、不要拷进 `src/`。
  - `hermes-agent/`：hermes-agent 源码，本项目记忆系统的主要参考对象。
  - `codex/`：OpenAI Codex CLI（Rust，`codex-rs`），参考其编码 agent 架构。
  - `knowledge-catalog/`：Google Cloud Knowledge Catalog（前身 Dataplex）示例，含 `okf/`（Open Knowledge Format + reference agent）、`toolbox/`、`samples/`，参考其上下文管理、富集与检索方案。
- `notes/hermes-memory-digest.md`：从下列 hermes 文档提炼的精读笔记，已按本项目目标筛过；平时优先看这份。

## 外部 Provider 范围

只适配两个，且都要本地模式：

| Provider  | 本地模式              | 说明 |
| --------- | --------------------- | ---- |
| Hindsight | `local_embedded`      | hermes 自带，跑本地 PostgreSQL daemon；接口直接参考其插件实现。 |
| Mem0      | OSS `Memory` 本地模式 | hermes 的 mem0 插件只支持云端（`MemoryClient` + API key）；本地化参考 `mem0` 官方库 OSS 模式（`from mem0 import Memory`，自配 LLM + 向量库），其插件的接口/线程模型仍可借鉴。 |

其余 6 个 provider（Honcho / OpenViking / Holographic / RetainDB / ByteRover / Supermemory）不在范围内。

## 参考阅读：hermes-agent 记忆系统文档

精要已整理进 `notes/hermes-memory-digest.md`，平时优先看；需原文再查下列文件（相对 `memory_repo/hermes-agent/`）：

- **核心设计**：`website/docs/user-guide/features/memory.md`（`MEMORY.md` 语义/容量/frozen snapshot）、`developer-guide/memory-provider-plugin.md`（`MemoryProvider` ABC、钩子、插件结构）、`developer-guide/prompt-assembly.md`（prompt 分层、缓存边界、记忆注入位置）、`developer-guide/agent-loop.md`（memory flush 时机）
- **Provider**：`plugins/memory/hindsight/` 与 `plugins/memory/mem0/` 的 `README.md` + `__init__.py`（接口/本地模式/线程模型）、`website/docs/user-guide/features/memory-providers.md`（只看 Hindsight、Mem0 两节）
- **参考手册**：`website/docs/reference/tools-reference.md`（`memory` toolset）、`user-guide/configuration.md`（Memory Configuration）、`reference/cli-commands.md`（`hermes memory` 子命令）
