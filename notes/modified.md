# 变更整理

## Tool / Model 调用层

- 原本只支持 bash tool，现在 tool-call 解析支持可配置的 `allowed_tools`，action 会携带 `tool_name`、`args`、`tool_call_id`，同时保留 `bash.command` 兼容旧环境执行。
- `litellm_model` / `litellm_response_model` 新增 `extra_tools` 注入机制，允许 agent 暴露 memory/provider tools；其他模型后端尚未完成该迁移。
- Responses API 增加 tool schema flatten、`function_call_output` 格式化、response output replay 处理，支撑 ModelHub / OpenAI Responses 风格调用。

## Memory

- 新增 `MemoryAgent`，注册为 `agent_class: memory`。
- 新增内置 memory tool，落盘到 `~/.mini-memory/MEMORY.md`，支持 `add` / `replace` / `remove`。
- Memory 注入采用 session-start frozen snapshot：本轮写入落盘，但只在下一轮/下一实例进入系统 prompt。
- 新增 `MemoryManager`，负责内置 memory、provider tool schema、tool 路由、session lifecycle、turn sync、shutdown。
- 新增 provider 抽象 `MemoryProvider`，当前支持单 external provider。
- 新增 Hindsight provider：仅 `local_embedded`，暴露 `hindsight_retain` / `hindsight_recall` / `hindsight_reflect`；自动 transcript 路径不再每个 tool step 写入，`sync_turn` 只 buffer，`on_session_end` 用 `retain_async=False` 同步写一次完整 task transcript，保证下一个 issue 开始前落入 Hindsight。
- 新增 Mem0 provider：OSS local Memory 模式，暴露 `mem0_search` / `mem0_note` / `mem0_observe`。
- 新增 `SessionStore`（SQLite + FTS5）：`MemoryManager.on_session_end` 把 transcript 写盘到 `~/.mini-memory/sessions.db`；同时注册内置 `session_search` tool（schema 与 `memory` 平级注入 `extra_tools`），支持 FTS5 查询语法（裸词、前缀 `*`、短语 `"..."`），按 bm25 排序、用 `snippet()` 高亮命中片段。整体不进 system prompt，不破坏 prefix cache。为保证最小 memory 实验口径清晰，`swebench_pro.yaml` 默认开启 sessions，把 `MEMORY.md + session_search` 作为最小 memory 设置。
- 新增 `consolidate_memory` + `ConsolidationConfig`：构造一个 memory-only prompt，跑一轮 `model.query`，**只**应用返回的 `tool_name == "memory"` actions（bash/session_search 一律 skip），调 `BuiltinMemory.add/replace/remove`；模型异常被吞，best-effort，不打断主循环。两个触发器：`consolidation.on_session_end`（`MemoryAgent.run` finally 触发，对应 hermes "session 结束跑一轮 LLM-only"）与 `consolidation.every_n_steps`（`MemoryAgent.step` override 中每 N 次模型调用触发，对应 hermes "session 过期前主动 flush" 的 mini-swe-agent checkpoint 等价物）。两个开关默认 OFF；`swebench_pro.yaml` 也默认关闭 consolidation，避免 simple `MEMORY.md` 实验混入后台总结变量。
- SWE-bench memory 配置统一进 `benchmarks/swebench_pro.yaml`（开启 `MemoryAgent`、注入 `memory_block`、内置 memory 块、`session_search`、Hindsight/Mem0 注释模板；默认实验口径为 `MEMORY.md + session_search`）。

## Chain-window 对照 baseline（独立路径，不进 memory 主线）

- 新增 `ChainWindowAgent`（`src/minisweagent/agents/chain_window.py`），`DefaultAgent` 子类，注册为 `agent_class: chain_window`。一条 SWE-bench-Pro 链共用同一 `self.messages`：第一题装 system + user，后续题只追加 user。每题前重置 `_task_n_calls` / `_task_cost`，**`step_limit` / `cost_limit` 按每题计**（`self.n_calls` / `self.cost` 仍累加全链供日志可见），否则长链后段题会因为继承前题计数立即触发 `LimitsExceeded`。
- 新增 `window_compress.py`：定义 `CompressionConfig`、`render_trace`、`extract_response_text`、`compress_history`。压缩 LLM 调用通过 `model.query_no_tools(messages, **kwargs)` 跑同主模型、不带 tools / tool_choice，避免 ModelHub 对 `tool_choice="none"` + `tools` 的 `UnsupportedParamsError`。
- `LitellmResponseModel` 新增 `query_no_tools`：剥掉 tool 相关 kwargs 的 Responses API 调用，专用于 compression 路径。
- 触发策略：每步执行后 + 进入新题前各检测一次 `usage.input_tokens >= model_window * threshold`。若超阈值且存在"系统 prompt 之后、当前题之前"的可压缩区段，把 `messages[1:_task_anchor]` 整体替换成一条 `<compressed_history>` user 消息；上一轮 summary 会被一起卷入下一次压缩，杜绝 summary 叠加超阈。
- 错误处理分两层：`query_no_tools` 自带 `retry()`（最多 20 次、10–60s 指数退避，复用 `LitellmModel.abort_exceptions`），吸收 RateLimit / 网络等瞬时错误；`_compress_now` 不再额外 cooldown，失败时只追加一条 `compression_log` 记录，下一次再触阈值会重试。
- `_seal_completed_task`：每题收尾时去掉 trailing `exit`，再为最后一次 assistant response 里没收到 `function_call_output` 的 tool call 补一条占位 output。否则下一题首个 `model.query` 会被 ModelHub 以 `-4003 No tool output found for function call` 拒掉。
- 新增 runner `mini-extra swebench-chain-window`（`src/minisweagent/run/benchmarks/swebench_chain_window.py`）：复用 `swebench.py` 的 dataset 加载 / chain 编排 / docker env / patch 提取 helper；每条链跑一个 `ChainWindowAgent`，per-instance 仍按原 layout 保存 traj。断点恢复按链做 all-or-nothing：完整链跳过；部分完成链清掉 preds + 各题 traj 目录，整条重跑（chain-window 上下文没法半恢复）。重启复用 `-o` 输出目录即可，日志会打印 `Resume: K chains fully done (skipped), M chains partially done ...`。
- 新增配置 `benchmarks/swebench_pro_chain_window.yaml`：默认 `model_window=272000`、`threshold=0.8`、`max_output_tokens=8192`、`char_budget=10000`、`trace_max_chars=600000`；系统 / 题目 prompt 跟 `swebench_pro_nomemory.yaml` 对齐，不加 chain-mode 额外说明，确保压缩 baseline 干净。
- 该路径不触发 `MemoryAgent` / `MemoryManager` / `SessionStore` / 任何 provider，完全独立于 memory 实验主线，可并行使用。

## SWE-bench / Harness Runner

- `swebench` runner 支持 `swe-bench-pro` subset，以及本地 `.json` / `.jsonl` 数据文件加载、`SWEBENCH_DATA_DIR`、`data/` 自动查找。
- SWE-bench-Pro 任务会把 requirements、interface 追加进 task。
- Docker image 解析支持 `image_name`、`docker_image`、SWE-bench-Pro 的 `dockerhub_tag`，并支持 `SWEAP_DOCKERHUB_USERNAME` 覆盖。
- Batch runner 不再固定 `DefaultAgent`，会读取 `agent.agent_class`，再动态包一层 progress tracking；同时传入 `session_id=instance_id` 给 memory。
- 环境配置改为按 instance 拷贝，避免并发 worker 修改共享 config。
- 新增按 instance regex 白名单转发 proxy 环境变量，避免全量污染容器环境。
- 启动环境后记录容器内初始 `HEAD`，用于后续干净抽 patch。
- Run 结束后从容器内 `git diff` 抽取最终 patch，包含小型 untracked 文件，排除 lockfile / metadata，并限制 patch 大小和文件数；抽取失败或非法时回退到 agent 自己提交的 patch。
- `swebench_single` 复用同一 dataset loader，也支持 `--agent-class`，并追加 Pro 的 requirements / interface。

## Docker 环境

- 新增 `api_timeout`，区分 Docker daemon 操作超时和命令执行超时。
- 新增 `container_entrypoint`，用于覆盖 SWE-bench-Pro 镜像里会干扰 `sleep` 的 `ENTRYPOINT`。
- 新增 `mem_limit`，传给 `docker run --memory/--memory-swap`。
- Docker 环境可接受 `docker://` / `oci://` 前缀并自动归一化。
- `cleanup` 的 stop/rm 超时改为使用 `api_timeout` / `pull_timeout`。

## ModelHub / Pro 配置入口

- 新增 `swebench_pro.yaml`：单一完整配置，覆盖 SWE-bench-Pro prompt、Docker 长超时/内存限制/`/app` 工作目录/source-only patch 规则；同时启用 `MemoryAgent` + 内置 `MEMORY.md`，并把 model 切到 `litellm_response`，绑定 ModelHub Responses endpoint、固定 `extra_headers`、`reasoning`、`max_output_tokens`、`store: false`、encrypted reasoning replay。
- 入口直接使用 `mini-extra swebench -c swebench_pro.yaml ...`，不再需要单独 overlay 或 wrapper。

## 依赖 / 测试

- `pyproject.toml` 新增可选依赖组：`hindsight`、`mem0`。
- 新增/扩展测试覆盖 memory manager/agent/provider、tool-call 解析、`litellm_model` / `litellm_response_model` extra tools、Docker 新配置、SWE-bench-Pro 数据/镜像/patch/proxy 行为。
- 新增 `tests/memory/test_session_store.py`、`tests/memory/test_consolidation.py`、`tests/memory/test_smoke.py`，外加在 `test_manager.py` / `test_integration.py` / `test_memory_agent.py` 中扩展覆盖：`SessionStore` 幂等写入与 FTS5 round-trip、`session_search` tool 路由 + 关闭路径 + 异常参数防御、`consolidation` action 过滤与 `max_actions` 截断、`every_n_steps` 触发节奏、跨 session 的 `MemoryAgent.run` → `session_search` 端到端流转、shipped `swebench_pro.yaml` 通过 `get_agent` 落地后行为符合预期。
- 另有 `data/smoke/swe_bench_pro_fast_10.txt`，用于记录一个 SWE-bench-Pro fast 10 实例子集。

