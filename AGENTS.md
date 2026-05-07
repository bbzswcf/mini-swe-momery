# AGENTS

## 项目目标

构建一套**面向代码任务（SWE-bench 类）**的轻量记忆系统（mini-memory）：让 agent 在解决 GitHub issue / repo-level 编码任务时，能跨 trial、跨 instance 复用之前学到的工程经验、踩过的坑、可行的改动模式。

明确的非目标（决定了我们不迁移哪些东西）：

- **不做用户画像** —— 没有"用户偏好/沟通风格"这种概念，整个系统只服务一个自动化 agent，因此不迁移 hermes 的 `USER.md`。
- **不做 messaging / multi-user / 权限** —— 不接 gateway、Telegram、Discord 这些。
- **不做 Skills（流程性知识）** —— 我们只做记忆（事实/经验性知识）。Skills 体系在 hermes 中是独立子系统，本项目完全不迁移。
- **不做多 profile / peer 建模** —— 单 agent 单实例足够。

## 目录约定

- `src/minisweagent/`：本项目的源码目录，所有新增代码写在这里。
- `memory_repo/hermes-agent/`：**只读参考仓库**。它是外部项目 hermes-agent 的源码，唯一作用是供我们参考其记忆系统的设计与实现。
  - 不要修改其中任何文件。
  - 不要将其作为依赖 import，也不要把其代码直接拷贝到 `src/` 中。
  - 仅用于阅读、对照实现思路。
- `notes/hermes-memory-digest.md`：从下列源文档提炼出的精读笔记，已按本项目目标筛过；优先看这份。

## 外部 Provider 范围

我们**只适配两个，且都要本地模式**：

| Provider  | 选用本地模式               | 说明                                                                                  |
| --------- | -------------------------- | ------------------------------------------------------------------------------------- |
| Hindsight | `local_embedded`           | hermes 自带这一模式，跑本地 PostgreSQL daemon；接口可直接参考其插件实现。             |
| Mem0      | OSS `Memory` 本地模式      | hermes 的 mem0 插件**只支持云端**（`MemoryClient` + API key），本地化时我们参考的是 `mem0` 官方库 OSS 模式（`from mem0 import Memory`，自配 LLM + 向量库）。其 plugin 的接口/线程模型仍可借鉴。 |

其他 6 个 provider（Honcho / OpenViking / Holographic / RetainDB / ByteRover / Supermemory）**不在本项目范围内**，相关文档无须深读。

## 参考阅读：hermes-agent 记忆系统相关文档

下列文档是构建本项目记忆系统时的主要参考来源。精要内容已整理在 `notes/hermes-memory-digest.md`，平时优先看那份。

### 一、核心设计（必读）

- `memory_repo/hermes-agent/website/docs/user-guide/features/memory.md` — 内置 `MEMORY.md` 的语义、容量、tool 行为、frozen snapshot 模式（USER.md 部分忽略）
- `memory_repo/hermes-agent/website/docs/developer-guide/memory-provider-plugin.md` — `MemoryProvider` ABC、生命周期钩子、插件目录结构
- `memory_repo/hermes-agent/website/docs/developer-guide/prompt-assembly.md` — 系统 prompt 分层组装、缓存边界、记忆注入位置
- `memory_repo/hermes-agent/website/docs/developer-guide/agent-loop.md` — agent loop 中 memory flush 的时机（compression / 每轮 / session 结束）

### 二、Provider 参考（仅这两个）

- `memory_repo/hermes-agent/plugins/memory/hindsight/README.md` + `__init__.py` — Hindsight plugin 的接口实现与本地模式配置
- `memory_repo/hermes-agent/plugins/memory/mem0/README.md` + `__init__.py` — Mem0 plugin 的接口实现（仅作生命周期/线程模型参考；其本身不支持本地）
- `memory_repo/hermes-agent/website/docs/user-guide/features/memory-providers.md` — 只看其中 Hindsight 和 Mem0 两节即可

### 三、参考手册中的 memory 章节

- `memory_repo/hermes-agent/website/docs/reference/tools-reference.md` — `memory` toolset 章节
- `memory_repo/hermes-agent/website/docs/user-guide/configuration.md` — `Memory Configuration` 章节
- `memory_repo/hermes-agent/website/docs/reference/cli-commands.md` — `hermes memory` 子命令（仅作 CLI 形态参考）
