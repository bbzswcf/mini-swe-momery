# hermes-agent 记忆系统精要笔记（mini-memory 视角）

> 整理自 `memory_repo/hermes-agent/` 中所有记忆相关文档与 plugin 源码。
> 本项目的目标是面向 **SWE-bench 类代码任务** 的轻量记忆系统，所以本笔记**已经裁剪**：
> - 不含用户画像（USER.md）相关内容
> - 不含 Skills、Honcho、用户建模、Messaging gateway、多 profile / peer 等不相关章节
> - 外部 provider 只覆盖 **Hindsight (local)** 和 **Mem0 (local)**

---

## 0. 一句话总览

hermes 的记忆系统由**两层并存**组成：

1. **内置层**：本地一份 markdown（`MEMORY.md`），以"frozen snapshot"形式注入 system prompt。
2. **外部 provider 层**：可插拔，单选。本项目只会接两个、且都跑本地：Hindsight `local_embedded`、Mem0 OSS。

底下还有一个**Session Search**：把所有历史会话存进 SQLite + FTS5，按需检索，不进 prompt。这套对 code 任务（跨 trial 复用经验）尤其有意义。

---

## 1. 内置记忆（Built-in）

> hermes 原本是双文件（MEMORY.md + USER.md）。本项目**只迁移 MEMORY.md**：服务对象不是人，没有"用户画像"这一层。下面也只整理 MEMORY.md 相关内容。

### 1.1 文件

| 文件        | 用途                                          | 字符上限              | 典型条目数 |
| ----------- | --------------------------------------------- | --------------------- | ---------- |
| `MEMORY.md` | agent 自己的工程笔记（环境/约定/教训/坑/经验）| 2,200（≈800 tokens）  | 8–15       |

存储位置：hermes 是 `~/.hermes/memories/`，我们落地时换成项目自己的路径（如 `~/.mini-memory/`）。

### 1.2 注入格式

session 启动时把磁盘内容渲染进 system prompt，长这样：

```
══════════════════════════════════════════════
MEMORY (your personal notes) [67% — 1,474/2,200 chars]
══════════════════════════════════════════════
Project under test: Django, PostgreSQL backend; runs on Python 3.11
§
pytest must be invoked from repo root with PYTHONPATH=. , otherwise tests import-fail
§
Known flaky tests: tests/integration/test_search.py — retry up to 3x
```

要点：

- 头部带 store 名 + 用量百分比（让 agent 自己感知容量）。
- 条目之间用 `§`（节号）分隔，支持多行。

### 1.3 Frozen Snapshot 模式（关键设计）

- system prompt 中的记忆块**只在 session 启动时拍一次快照**，session 内绝不变。
- session 内 agent 调 `memory` tool 写入会**立即落盘**，但**不会进当前 system prompt**，要等下一个 session。
- tool 响应**永远显示当前最新状态**，所以 agent 自己知道写了什么。

> 这样做是为了保住 LLM provider 的 **prefix cache**，避免每次写记忆都让前缀失效。在 SWE-bench 这类长循环 agent 上，缓存命中率直接决定成本。

### 1.4 `memory` tool 的语义

只有 3 个 action（**没有 read**，因为已经注入 prompt）：

| action    | 说明                                                                  |
| --------- | --------------------------------------------------------------------- |
| `add`     | 新增条目                                                              |
| `replace` | 通过 `old_text` 子串匹配定位旧条目，整体替换为新内容                  |
| `remove`  | 通过 `old_text` 子串匹配删除条目                                      |

**子串匹配规则**：`old_text` 只需是唯一识别条目的一段子串，不必是全文；如果命中多条则报错要求更具体。

> 我们裁掉了 `target` 参数（hermes 用它在 memory/user 间切换）；只剩一个 store。

### 1.5 容量管理

写入超限时返回结构化错误：

```json
{
  "success": false,
  "error": "Memory at 2,100/2,200 chars. Adding this entry (250 chars) would exceed the limit. Replace or remove existing entries first.",
  "current_entries": ["..."],
  "usage": "2,100/2,200"
}
```

约定的处理流程：读现有条目 → 找可合并/可删 → 用 `replace` 合并精简 → 再 `add`。最佳实践：用量 >80% 就开始主动合并。

### 1.6 其他约束

- **去重**：精确重复的条目自动拒绝（返回 success + "no duplicate added"）。
- **安全扫描**：写入前扫描注入/外泄模式（prompt injection、credential exfil、不可见 unicode 等），命中即拒。
  - 对 SWE-bench 场景这一项可适度简化，但**不可见 unicode 检测必须保留**——agent 抓 issue 描述很容易把奇怪字符也吞进来。
- **配置**：

  ```yaml
  memory:
    enabled: true
    char_limit: 2200
  ```

  比 hermes 的配置少了 `user_profile_enabled` 和 `user_char_limit`。

### 1.7 什么该写、什么不写（针对代码任务的版本）

| 应写                                                            | 不写                            |
| --------------------------------------------------------------- | ------------------------------- |
| 项目环境（"Django 4.2 + PG 15，runtime Py3.11"）                 | 一句话过于含糊（"修了 bug"）    |
| 构建/测试坑（"pytest 必须从 repo root 跑，否则 import 失败"）    | 容易再次查到的事实              |
| 修复模式（"`Manager.objects.bulk_create` 在 SQLite 下会丢 PK"） | 大段日志/diff dump              |
| Repo 约定（"用 black + ruff，行宽 100"）                        | session 临时文件路径等          |
| 已 verify 但失败过的方案（避免下次重蹈覆辙）                    | issue 原文（已经在 input 里）   |
| 跨 trial 仍然成立的事实                                         |                                 |

**条目风格**：紧凑、信息密度高、可执行。每条最好把多个相关事实打包到一起，而不是分条记。

---

## 2. Session Search（与 memory 互补，对 code 任务尤其有用）

| 维度           | Persistent Memory          | Session Search                |
| -------------- | -------------------------- | ----------------------------- |
| 容量           | ~800 tokens 总计           | 无限（所有历史 session）      |
| 速度           | 即时（在 prompt 里）       | 搜索 + LLM 总结               |
| 用途           | 关键事实长期在场           | "上次跑这个 instance 时怎么解决的？" |
| 维护           | agent 手动 curate          | 自动——所有 session 全部入库   |
| token 成本     | 每 session 固定 ~800       | 按需（搜的时候才花）          |

实现：所有 session 存入 SQLite + FTS5 全文索引，提供一个 `session_search` tool 按需查询，搭配一个轻量 LLM 做摘要。

> 对 SWE-bench 来说，这一层正好对应"我之前跑过同一个 repo 的别的 issue 吗？"的能力；可以考虑独立做掉、不必非得绑在 provider 上。

**已迁移到 mini-memory**（独立于任何 provider）：

- `src/minisweagent/memory/session_store.py`：薄封装的 SQLite + FTS5；`messages_fts` 索引每条消息，`sessions` 表存元数据，`record_session` 对同一 `session_id` 幂等（先 DELETE 再 INSERT）。
- `MemoryManager` 默认开 `sessions_enabled=True`，`SessionStore` 落在 `home / "sessions.db"`；可在 yaml 用 `sessions_enabled: false` 关掉。
- `session_search` 作为内置 tool（schema 与 `memory` 平级注册到 `extra_tools`），返回 `{session_id, role, snippet, summary, ...}` 列表，支持 FTS5 查询语法。
- `MemoryAgent.run()` finally 路径中调 `manager.on_session_end(messages, model=...)`，自动写盘。
- 暂未实现 LLM 摘要：`sessions.summary` 字段保留但 `record_session` 默认写空字符串，后续若启用 consolidation 也只写 MEMORY.md 不回填 summary。

---

## 3. 系统 Prompt 的分层组装（cache 友好）

> 已删去 SOUL.md / USER profile / Skills index / Honcho / messaging platform 这些与本项目无关的层。

两个核心思想：

1. **缓存层 vs API call 时层** 分离，保住 prefix 稳定。
2. **mid-session 不改 system prompt**，记忆/上下文变更走"下一 session 生效"或"在当前 turn 的 user message 里 overlay"。

### 3.1 我们关心的缓存层（精简版）

```
1. agent identity / 行为指引
2. memory provider 的静态块  (active 时；prefetch 之外的固定信息)
3. frozen MEMORY snapshot
4. (可选) 项目上下文文件     (AGENTS.md / repo 内的工程约定)
5. timestamp + session id
```

### 3.2 不进缓存的层（每次 API call 时才拼）

- `ephemeral_system_prompt`
- prefill messages
- **provider 的 prefetch recall**：注入到**当前 turn 的 user message**，而不是 system prompt——这是为了不污染缓存前缀

### 3.3 设计动机

- 保 provider 端 prompt cache
- 不无谓 mutate 历史
- 让 memory 语义可理解（"这一 session 内你看到的记忆 = session 启动时的快照"）

---

## 4. Agent Loop 中的记忆时机

来源：`agent-loop.md` 中的 memory 提及点。

- 每轮文本响应**返回前**会"flush memory if needed"
- 每轮结束：messages 写 SQLite（session 存储）；memory 改动写到 `MEMORY.md`
- **context compression 第一步就是先 flush memory**——避免摘要过程中丢失尚未落盘的记忆变更
- session resume 后从 SQLite 恢复

> 这条 "compression 之前先 flush" 在 SWE-bench 长循环里特别重要：长 trace 一定会触发压缩，必须保证压缩前把 agent 学到的事实先固化下来。

### 4.3 mini-memory 的实际接线（与 hermes 的差异点）

我们落地时**只接通**了下列钩子，并刻意**砍掉**了一部分 hermes 的自动行为：

| hermes 钩子                   | mini-memory                 | 原因                                                                                                  |
| ----------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| `initialize(session_id, ...)` | ✅ wired                      | 每次 `agent.run` 入口 + finally 路径上一定执行                                                         |
| `system_prompt_block()`       | ✅ wired                      | 注入 system prompt（builtin snapshot + provider 块）                                                   |
| `get_tool_schemas()`          | ✅ wired → `model.extra_tools` | `MemoryAgent.__init__` 里推到 model 的 `extra_tools`，这样 LLM 才看得见 `memory` / `mem0_*` / `hindsight_*` |
| `handle_tool_call(name,args)` | ✅ wired                      | `MemoryAgent.execute_actions` 通过 `manager.tool_names` 路由                                          |
| `sync_turn(user,assistant)`   | ✅ wired                      | 每轮在 `add_messages` 之前调，避免把刚生成的 observation 误当作 "user side"                            |
| `on_memory_write(action,content)` | ✅ wired                  | `MemoryManager._handle_builtin` 在 builtin 写成功后转发                                                |
| `on_session_end(messages)`    | ✅ wired                      | `MemoryAgent.run` 的 finally 里**始终**执行（即使 `run` 抛异常）                                       |
| `shutdown()`                  | ✅ wired                      | 仅当 manager 由 agent 自己构造时才调（外部 manager 由调用方负责）                                       |
| `prefetch(query)`             | ❌ 已删                        | hermes 把 prefetch 文本注入到当前 turn 的 user message —— mini-memory 走 prompt-cache-friendly 路线，禁止 mid-session 改 prompt（§3.3）。recall 改为 model 主动调 `*_recall` / `mem0_search`。 |
| `queue_prefetch(query)`       | ❌ 已删                        | 同上：没有消费方就没有意义                                                                            |
| `on_pre_compress(messages)`   | ❌ 已删                        | mini-swe-agent 没有 context compression                                                                |

---

## 5. 外部 Memory Provider 体系

### 5.1 单 Provider 规则

> 同一时刻只能有一个外部 provider 激活；内置层始终激活；试图注册第二个会被 `MemoryManager` 拒绝。

理由：避免 tool schema 膨胀和后端冲突。本项目继承这个规则。

### 5.2 激活后系统自动做的 6 件事

1. 把 provider 静态块**注入 system prompt**（provider 知道什么）
2. **每个 turn 之前 prefetch**（Hermes 原设计：后台、非阻塞，结果叠到当前 turn 的 user message；mini-memory 已删）
3. **每个 turn 之后 sync** 对话（mini-memory 仍接线；Hindsight 只本地 buffer，Mem0 可后台写）
4. **session 结束时提取记忆**（如 provider 支持）
5. 把内置 memory 的写**镜像**到外部 provider（可选，看是否真的需要）
6. 注入 provider 自带的 tools（搜索 / 存储 / 管理）

### 5.3 我们要支持的 provider（仅 2 个，都本地）

| Provider     | 模式                  | 本地依赖                                                          | 独特卖点                                |
| ------------ | --------------------- | ----------------------------------------------------------------- | --------------------------------------- |
| Hindsight    | `local_embedded`      | 内嵌 PostgreSQL daemon（hermes 已实现）；只需一个 LLM API key | 知识图谱 + 实体抽取 + 跨记忆 reflect 综合 |
| Mem0 (OSS)   | `from mem0 import Memory` | 一个 LLM、一个向量库（默认 Qdrant 或 Chroma），全跑本地        | server 端 LLM 自动抽取记忆 + 自动去重   |

详见 §8、§9。

---

## 6. 写一个 Memory Provider Plugin（最关键的设计抽象）

来源：`developer-guide/memory-provider-plugin.md`。这是我们要复刻的核心抽象。

### 6.1 目录结构

```
plugins/memory/<name>/
├── __init__.py      # MemoryProvider 实现 + register() 入口
├── plugin.yaml      # 元数据（name, description, hooks）
└── README.md        # 安装/配置/工具说明
```

> 我们裁掉了 `cli.py` —— 本项目暂不做插件级 CLI 子命令。

### 6.2 `MemoryProvider` ABC

#### 必实现（mini-memory 版）

| 成员                              | 何时调用                           |
| --------------------------------- | ---------------------------------- |
| `name` (property)                 | always                             |
| `is_available()`                  | agent init / 激活前；**禁止网络** |
| `initialize(session_id, *, home, **kwargs)` | agent 启动一次；`home` 是 per-instance 状态目录；额外 kwargs 由 manager 透传，provider 应忽略未知参数 |
| `get_tool_schemas()`              | tool 注入；schema **必须**是 OpenAI tool-call wrapper 格式（`{"type": "function", "function": {...}}`），与 `BASH_TOOL` 同级别 |

`handle_tool_call(name, args)` 在 mini-memory 里改成**非 abstract**（默认抛 not-handled 错误），这样 context-only provider 不必强制实现。

> 我们去掉了 hermes 的 `get_config_schema` / `save_config`：本项目没有 setup wizard，配置直接通过 `MemoryManagerConfig` / yaml 走。

#### 可选钩子（生命周期）

| 钩子                                  | 时机                | mini-memory 接入                                                                  |
| ------------------------------------- | ------------------- | --------------------------------------------------------------------------------- |
| `system_prompt_block()`               | system prompt 组装时 | ✅                                                                                 |
| `sync_turn(user, assistant)`          | 每轮完成            | ✅；provider-specific。Hindsight 只 buffer，任务结束同步写一次；Mem0 仍用后台单写线程 |
| `on_session_end(messages)`            | 会话结束            | ✅；MemoryAgent.run 的 finally 路径里调用                                           |
| `on_memory_write(action, content)`    | 内置 memory 被写入时 | ✅；镜像到 provider 后端                                                            |
| `shutdown()`                          | 进程退出            | ✅；仅由 owns-manager 的 agent 自己调                                               |
| ~~`prefetch(query)`~~                 | 每个 API call 之前  | ❌ 已从 ABC 删掉：会破 prefix cache（详 §4.3）                                       |
| ~~`queue_prefetch(query)`~~           | 每轮结束后          | ❌ 已删                                                                            |
| ~~`on_pre_compress(messages)`~~       | context 压缩之前    | ❌ 已删（mini-swe-agent 没有 compression）                                          |

### 6.3 配置 Schema

```python
def get_config_schema(self):
    return [
        {
            "key": "api_key",
            "description": "LLM API key (for memory extraction/synthesis)",
            "secret": True,
            "required": True,
            "env_var": "MINI_MEM_LLM_API_KEY",
        },
        {"key": "mode", "default": "local", "choices": ["local"]},
    ]
```

- `secret: True` + `env_var` → 写到 `.env`
- 非 secret 字段 → 交给 `save_config()` 写到 native 位置（如 `<home>/<name>.json`）
- **建议**：setup 向导只问"必须配置的"（API key 等），其他高级选项放到 native 配置文件中由用户手编。

### 6.4 三条硬约束

1. **热路径 `sync_turn()` 必须便宜**。低成本 provider 可以在这里后台写；Hindsight 这类 retain 可能触发 LLM extraction 的 provider 不应每轮写，只在这里 buffer，等 `on_session_end()` 同步写一次：

   ```python
   def sync_turn(self, user_content, assistant_content):
       self._turns.append({"user": user_content, "assistant": assistant_content})

   def on_session_end(self, messages):
       self._api.retain_batch(self._turns, retain_async=False)
   ```

2. **路径隔离**：所有路径必须用 `initialize` 传入的 `home`，**不可** hardcode `~/.mini-memory`，否则跨 instance 数据互窜。

3. **单 provider**：见 §5.1。

### 6.5 入口

```python
# __init__.py
def register(ctx) -> None:
    ctx.register_memory_provider(MyMemoryProvider())
```

```yaml
# plugin.yaml
name: my-provider
version: 1.0.0
description: "Short description."
hooks:
  - on_session_end
```

### 6.6 Mem0 plugin 中值得照搬的工程实践

`memory_repo/hermes-agent/plugins/memory/mem0/__init__.py` 自带几条很值得借鉴的工程细节（即使其本身只支持云端）：

- **Lazy + thread-safe client**：`_client_lock` 包住懒初始化，避免并发首次调用时多重创建。
- **Circuit breaker**：连续 N 次失败后开断路器，冷却 K 秒——防止后端挂了的时候疯狂重试拖慢主流程。
- **`_read_filters` vs `_write_filters` 分离**：读的时候只用 user_id，写的时候带上 agent_id，便于归因。
- **prefetch 用 `_prefetch_lock` + 单字符串缓冲**：上轮的 prefetch 结果在下次 `prefetch()` 调用时被原子取走 + 清空；同时只允许一个 prefetch 线程在跑，新的会先 `join(timeout=...)` 旧的。
- **prefetch / sync 结果都用一致的 `_record_success` / `_record_failure`** 喂给断路器。

这五条全部 generalize，无关 Mem0 本身。

---

## 7. Memory Flush 生命周期（来自 Gateway）

来源：`gateway-internals.md` 的 Memory 两节。本项目不接 gateway，但 flush 时机的设计仍然适用：

### 当 session 重置 / 恢复 / 过期时

1. 内置 memory flush 到磁盘
2. 触发 provider 的 `on_session_end()`
3. **跑一个临时 agent 做 memory-only 的一轮对话**（让 LLM 自己整理记忆）
4. 上下文丢弃或归档

### 后台维护

- session 过期前**主动 flush memory**
- 与压缩/缓存刷新等并行

> 关键洞见："让 LLM 自己用一轮对话整理记忆"是个简单但有效的模式。在 SWE-bench 的场景下，这一轮可以是"基于刚刚 trace，提取本次最值得记的 1–3 条事实，必要时 replace 已有条目"。

**已迁移到 mini-memory**：

- `src/minisweagent/memory/consolidation.py` 提供 `consolidate_memory(model, builtin, messages, *, max_actions, summary_max_chars)`：构造一个 memory-only prompt，调用 `model.query(...)` 一次，只解析返回的 `tool_name == "memory"` actions（其它 tool call 一律 skip），再以 best-effort 方式调 `builtin.add/replace/remove`。整个函数 try/except 包裹模型调用，**永远不抛**。
- `MemoryManagerConfig.consolidation: ConsolidationConfig` 引入两个触发器：
  - `on_session_end: bool`：`MemoryAgent.run()` finally 时调用一次（对应 hermes "session 重置/恢复/过期时" 的第 3 步）。
  - `every_n_steps: int`：`MemoryAgent.step()` override 里每 N 次模型调用触发一次（对应 hermes "后台主动 flush memory"——mini-swe-agent 没有 session 过期，但等价于"长 trace 的 checkpoint flush"，把当下学到的事实先落盘以防中途 OOM/超时）。
  - 两个开关默认全部 OFF，保证不增加意外的 LLM 成本；`swebench_pro.yaml` 也默认关闭 consolidation，用于先测最小 `MEMORY.md + session_search` 实验。
- 关键约束：consolidation 用的是独立的临时 prompt，不动 `self.messages`，因此 **不破坏 frozen snapshot / prefix cache**（写盘只在下一 session 注入 system prompt）。

---

## 8. Hindsight：本地模式参考

源文档：`memory_repo/hermes-agent/plugins/memory/hindsight/README.md` 与 `website/docs/user-guide/features/memory-providers.md` 的 Hindsight 节。

### 8.1 三种模式中我们只看 `local_embedded`

| 模式             | 含义                                                              |
| ---------------- | ----------------------------------------------------------------- |
| `cloud`          | 走 Hindsight Cloud API。**不用**。                                |
| `local_embedded` | hermes 启一个本地 Hindsight daemon，内置 PostgreSQL；后台启动，5 min 不活跃自动停。**用这个**。 |
| `local_external` | 接已有的 Hindsight 实例（自己 docker 起的）。备选。               |

### 8.2 本地模式依赖

- 任意一个 LLM API key（OpenAI / Anthropic / Gemini / Groq / OpenRouter / MiniMax / Ollama / 任何 OpenAI-compatible 端点）——用于"记忆抽取"和"reflect 合成"
- embedding 和 reranking 跑本地，无须额外 key
- 通过 `HINDSIGHT_LLM_API_KEY` 配 LLM key
- 日志：`~/.hermes/logs/hindsight-embed.log`

### 8.3 关键配置项（本地模式相关）

| key                | 默认值                                  | 含义                                     |
| ------------------ | --------------------------------------- | ---------------------------------------- |
| `mode`             | `cloud` → 我们改 `local_embedded`       | 模式                                     |
| `bank_id`          | `hermes` → 我们改 `mini-memory`         | 记忆 bank 名                             |
| `bank_id_template` | —                                       | 动态生成 bank 名（`{profile}` 等占位符）|
| `bank_mission`     | —                                       | reflect 时的身份/视角设定                |
| `bank_retain_mission` | —                                    | 控制"该提取什么"                         |
| `recall_budget`    | `mid`                                   | 回忆深度：`low` / `mid` / `high`         |
| `recall_prefetch_method` | `recall`                          | 自动 prefetch 走原始 facts 还是 LLM 合成 |
| `auto_recall`      | `true`                                  | 每 turn 之前自动 recall                  |
| `auto_retain`      | `true`                                  | 每 turn 之后自动 retain                  |
| `retain_async`     | `true`                                  | retain 异步                              |
| `retain_every_n_turns` | `1`                                 | 每 N turn retain 一次                    |
| `memory_mode`      | `hybrid`                                | `hybrid` / `context` / `tools`           |
| `llm_provider`     | `openai`                                | 本地 LLM 提供方                          |
| `llm_model`        | per-provider                            | 模型名                                   |
| `llm_base_url`     | —                                       | `openai_compatible` 时的端点 URL         |

`memory_mode`：
- `hybrid` —— 自动注入 + 暴露 tool（默认）
- `context` —— 仅自动注入，不暴露 tool
- `tools` —— 仅 tool，不自动注入

### 8.4 工具集（在 hybrid / tools 模式下暴露）

| Tool                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| `hindsight_retain`  | 存储信息，自动抽实体，可带 `tags`                            |
| `hindsight_recall`  | 多策略检索（语义 + 实体图）                                  |
| `hindsight_reflect` | 跨记忆 LLM 合成（**别的 provider 没有的能力**）             |

### 8.5 给 mini-memory 的启示

- 抽 LLM 出来作为 provider 配置的一等公民——抽取/合成都要 LLM
- "**reflect**" 这个动作（跨多条记忆做 LLM 合成）值得借鉴：是 retain 和 recall 之外的第三档能力，特别适合"我之前在类似 issue 中得到过哪些一致的结论"这类查询
- bank_id 模板化（按 profile / repo 派生）是个简单实用的隔离手段
- `retain_every_n_turns` 提供了一个 "不必每 turn 都灌" 的简单旋钮，省 token

---

## 9. Mem0：本地模式参考

源文档：`memory_repo/hermes-agent/plugins/memory/mem0/README.md` + `__init__.py`，以及 mem0 OSS 文档（`https://docs.mem0.ai`，外部）。

### 9.1 重要前提

> hermes 自带的 mem0 plugin **只支持云端**：它内部用的是 `from mem0 import MemoryClient` + `MEM0_API_KEY` 调 Mem0 Platform。
>
> 我们的本地模式要走 mem0 OSS 的另一个入口：`from mem0 import Memory`，自配 LLM 和向量库。**hermes 的实现不能直接用作本地模板**，但它的 plugin 骨架（lazy client / circuit breaker / prefetch / sync_turn 三件套）可以原样借鉴。

### 9.2 OSS 本地模式依赖（Mem0 自身）

- 一个本地或托管的 LLM（用于事实抽取）
- 一个向量库（默认 Qdrant，可换 Chroma / pgvector / FAISS 等）
- `pip install mem0ai`

### 9.3 Mem0 提供的核心动作（OSS 与云端 API 一致）

| API                            | 行为                                            |
| ------------------------------ | ----------------------------------------------- |
| `m.add(messages, user_id=...)` | 服务器端 LLM 抽取事实，自动去重；可关 `infer=False` 存 raw |
| `m.search(query, user_id=...)` | 语义检索 + 可选 rerank                          |
| `m.get_all(user_id=...)`       | 取此 user 全部记忆（用于 profile 工具）        |
| `m.delete(memory_id=...)`      | 按 id 删                                        |

hermes 的 plugin 只暴露三个 tool（`mem0_profile` / `mem0_search` / `mem0_conclude`），分别对应 `get_all` / `search` / `add(infer=False)`。这个 tool 集合很合适本项目复用。

### 9.4 给 mini-memory 的启示

- **`infer=True/False` 双模式**：默认让 LLM 抽取（agent 不用费心写好），关键时刻直接 `infer=False` 存原文。这种"动作即语义"的 tool 设计很值得抄（参考：hermes 命名为 `conclude`）。
- **scoped filters 分读写**（见 §6.6）：读不带 agent_id 让你跨 agent 看到所有记忆，写带 agent_id 便于归因。在本项目里，"agent_id" 可换成 "trial_id" 或 "instance_id"。
- **断路器对外部组件的保护**：Mem0 OSS 跑本地，向量库或 LLM 偶尔抽风时一样需要断路器，照抄即可。

---

## 10. 给 mini-memory 的设计启示（提炼）

读完这套（已剔除非目标内容），下面是我们做自己的记忆系统时**最值得保留**的设计点：

1. **"内置 + 可插拔外部"分层**。永远保留一个零依赖、可控、有上限的内置层；外部接口走 provider 抽象，单选。
2. **Frozen Snapshot**。session 内不改 system prompt，写盘即时但 prompt 内容直到下个 session 才更新——保住 LLM 的 prefix cache。SWE-bench 长 trace 里这是省钱的关键。
3. **Tool 只暴露 add/replace/remove**，没有 read：节省 tool call、避免 agent 重复读自己已经看到的内容。
4. **子串匹配定位条目**：减少 token 消耗，避免要求 agent 复述整段。
5. **强制字符上限 + 用量百分比写在 prompt header**：让 LLM 自己感知压力、自主合并。
6. **超限错误结构化返回 + 引导合并**：错误不是终点，是工作流的一步。
7. **写入前安全扫描**：注入/外泄/隐藏 unicode。SWE-bench 里 issue 描述往往是用户提供的，会带奇怪字符，必须扫。
8. **Provider 接口最小集**：`name / is_available / initialize / get_tool_schemas`；`handle_tool_call` 给一个 default-error，允许 context-only provider 不实现。其余皆为可选钩子。
9. **`sync_turn` 不能做重活**。Mem0 这类本地写入可用单写线程后台化；Hindsight 这类可能触发 LLM extraction 的 provider 改为每轮只 buffer，在 `on_session_end` 用 `retain_async=False` 同步写完整 task transcript，确保链式实验进入下一个 issue 前记忆可见。
10. ~~`on_pre_compress` 钩子~~：mini-swe-agent 无 compression 流程，已从 ABC 里删；同等需求在 `on_session_end` 里处理。
11. **路径隔离**：所有存储路径从一个统一的"home"参数派生，绝不 hardcode。`MemoryManager` 现在会断言 `BuiltinMemory.path.parent == home`，避免 builtin 与 provider 状态目录漂移。
12. **Session 颗粒度可配**：在 SWE-bench 上至少要有 `per-instance` 和 `per-repo` 两档。
13. **Provider 内的工程模板**（来自 hermes mem0 plugin）：lazy thread-safe client、circuit breaker、读写 filter 分离、统一喂断路器。  
    **mini-memory 修订**：~~prefetch 单缓冲 + 单线程~~ 已废弃——recall 只走 model 主动调 tool 的路线，否则会破 prefix cache（§3.3）。  
    **断路器规则**：`is_available()` 返回 False（如 `mem0` 未安装）时**直接返回 not-available 错误，不经断路器**，否则重装后断路器仍是 stuck-open。
14. **`infer=True/False` 双语义入口**（来自 mem0 OSS）：mini-memory 约定 `sync_turn` 用 `infer=False`（保留原文），`mem0_observe` 是**唯一**的 fact-extraction 入口（`infer=True`），避免双重抽取。`mem0_note` 同样 `infer=False`，存 verbatim。
15. **跨记忆 "reflect" 能力**（来自 Hindsight）：在 retain/recall 之外的第三档——"基于已有记忆做 LLM 合成"。SWE-bench 中适合"我之前在类似 issue 中验证过哪些事？"这类提问。
16. ~~Compression 之前先 flush memory~~：mini-swe-agent 无 compression，跳过。
17. **session 结束让 LLM 跑一轮"memory-only"对话整理记忆**：简单但有效，比"靠每 turn 自己规整"更稳。
18. **Tool schema 必须真的送达模型**：mini-swe-agent 的 model layer 原本 hardcode `tools=[BASH_TOOL]`。mini-memory 的接线办法是 model 类提供 `extra_tools` 属性，`MemoryAgent` 在 `__init__` 推 `manager.get_tool_schemas()` 进去；parser 接收 `allowed_tools` 名单。  
    解析后的 action 一律带 `tool_name`（bash 兼容性继续带 `command`），`execute_actions` 据此路由 tool → manager / env，避免"测试用 mock 通过、真模型整套静默失效"的死路。
=[BASH_TOOL]`。mini-memory 的接线办法是 model 类增加 `extra_tools` 属性，`MemoryAgent` 在 `__init__` 把 `manager.get_tool_schemas()` 推进去；parser 也接收 `allowed_tools` 名单，不再硬要求 name == "bash"。  
    解析后的 action 一律带 `tool_name` 字段（bash 为兼容仍带 `command`），`execute_actions` 据此路由 tool → manager / env，避免"测试用 mock 通过、真模型整套静默失效"的死路。
型整套静默失效"的死路。
