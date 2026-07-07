# Mini-Memory Reproduction Spec (Agent-Native, Agent-Free)

This document distills the **mini-memory** memory subsystem into a self-contained
reproduction spec. The goal is that *another agent*, given only this document,
can re-implement the same memory behavior on top of *any* tool-using LLM agent
loop, in any language, without reading our source tree.

It is **agent-free**: the memory machinery is defined purely in terms of a small
"Host Contract" (§1). Anything your agent base already does — model calls, tool
dispatch, prompt assembly — is abstracted behind that contract. It is
**agent-native**: every interface is expressed as data shapes, tool JSON
schemas, and prompt strings that an LLM agent can consume directly.

Two methods are specified, both targeting **code tasks (SWE-bench-style)** where
an agent repeatedly solves GitHub-issue-like problems and should reuse hard-won
engineering knowledge across attempts:

- **Method A — `MEMORY.md` + `session_search`**: a bounded distilled-notes file
  injected into the prompt, plus a local full-text index over past transcripts
  the agent can query on demand. *(the default, minimal setup)*
- **Method B — Filesystem chain memory**: a per-chain directory of Markdown
  files the agent reads/searches with plain shell commands; distilled notes are
  written by a memory-only LLM pass at task end, raw evidence by the host.

Out of scope (intentionally): user profiles, multi-user/messaging, "skills"
(procedural knowledge), vector DBs / embeddings, and external SaaS memory
providers. Those are separate ablations and not part of this reproduction.

---

## 1. The Host Contract (what your agent must provide)

The memory subsystem is a **Memory Manager** object that your agent loop calls at
five well-defined points. Implement these five hooks and you can reproduce either
method without coupling to our agent classes.

### 1.1 Concepts and vocabulary

- **Session**: one task attempt = one solve of one problem instance (one
  SWE-bench instance / one trial). It has a stable string `session_id`
  (use the instance id).
- **Chain** *(Method B and chained experiments)*: an ordered list of related
  sessions that should share memory (e.g. several issues on the same repo).
  Identified by `chain_id`; each session within carries a `step_index`
  (0-based order). Different chains are fully independent and get **separate
  memory homes**.
- **Transcript / `messages`**: the ordered list of message dicts the agent
  accumulated during a session (system, user, assistant, tool outputs). The
  exact shape your loop uses is fine as long as text can be extracted (§1.4).
- **Model**: an object exposing `model.query(messages) -> response`. Optionally
  `model.query_no_tools(messages) -> response` (same model, no tools attached) —
  used by the memory-only passes so they don't pollute or depend on the main
  tool set. If absent, fall back to `model.query`.

### 1.2 The five lifecycle hooks

| Hook | When the agent calls it | What it does |
| --- | --- | --- |
| `initialize(session_id, **ctx)` | at session start, before the first model call | refresh in-prompt state; `ctx` may carry `chain_id`, `step_index`, etc. |
| `system_prompt_block() -> str` | while assembling the system prompt | returns the text block to inject (frozen MEMORY snapshot and/or filesystem policy) |
| `get_tool_schemas() -> list[dict]` | once, when registering tools with the model | returns OpenAI-style function tool schemas to expose alongside `bash` |
| `handle_tool_call(name, args) -> dict` | whenever the model calls a memory tool | executes the tool and returns a JSON-serializable result dict |
| `on_session_end(messages, *, model=None)` | in a `finally` after the session ends (success, error, or limit) | persists distilled/raw memory for future sessions |

Optional sixth hook for the consolidation ablation (§2.5):
`maybe_consolidate(model, messages, *, n_calls) -> dict | None`, called after
each model step.

### 1.3 Integration glue (agent-free pseudocode)

```python
# At construction: expose memory tools to the model next to your own tools.
model.tools += manager.get_tool_schemas()

# At run start:
def run(task, session_id, **ctx):
    manager.initialize(session_id, **ctx)
    system_prompt = render(system_template, memory_block=manager.system_prompt_block())
    try:
        loop:                                   # your normal agent loop
            response = model.query(messages)
            for call in response.tool_calls:
                if call.name in manager.tool_names:
                    out = manager.handle_tool_call(call.name, call.args)  # JSON dict
                else:
                    out = env.execute(call)                                # bash etc.
                append_observation(messages, out)
            # optional: manager.maybe_consolidate(model, messages, n_calls=step_count)
    finally:
        manager.on_session_end(messages, model=model)
```

Two routing rules make this agent-free:

1. **Tool routing by name.** `manager.tool_names` is the set of names the
   manager owns (`{"memory", "session_search", ...}`). Any call whose name is in
   that set goes to `handle_tool_call`; everything else (notably `bash`) goes to
   your environment unchanged.
2. **Memory tool results are transport-success.** A memory tool call always
   "succeeds" at the protocol level; failure (capacity, no match) is encoded
   inside the returned JSON via `success`/`error` keys. When you format the
   observation back to the model, emit it as a normal successful tool output
   (e.g. `returncode=0`, no exception) so the model reads the JSON rather than
   treating it as a shell crash.

### 1.4 Message-text extraction (`extract_message_text`)

Both methods flatten a message dict into "the text the LLM actually saw." This
is the only place that touches your transcript shape; adapt the field names to
your loop. Reference behavior:

```python
def extract_message_text(msg) -> str:
    parts = []
    add_content(parts, msg.get("content"))            # str | [ {text|output_text|input_text} ]
    add_response_output(parts, msg.get("output"))     # Responses API: list of items
    if msg.get("type") == "function_call_output":
        add_content(parts, msg.get("output"))
    if msg.get("tool_name"):
        parts.append("tool:" + msg["tool_name"])
    if msg.get("tool_calls"):
        parts.append(json_dumps({"tool_calls": msg["tool_calls"]}))
    for action in (msg.get("extra") or {}).get("actions", []):
        parts.append(json_dumps({"tool": action.get("tool_name", "bash"),
                                 "args": action.get("args") or {}}))
    return "\n".join(p for p in parts if p).strip()
```

`add_response_output` handles Responses-API items: `type=="function_call"` →
`{tool, args}`; `type=="function_call_output"` → its `output`; otherwise its
`content`.

**Critical invariant:** never index or summarize off-band debug fields such as
`extra.raw_output` (the untruncated bash dump kept only for replay). Indexing it
blows past API content limits and pollutes recall with text the model never saw.
Only flatten what was actually in the model's context.

---

## 2. Method A — `MEMORY.md` + `session_search`

This is the default, minimal memory setup. It exposes **two tools** to the agent
and adds **one block** to the system prompt.

- `MEMORY.md`: a small, bounded file of *distilled* durable knowledge, **injected
  into the system prompt** so it is always present (a frozen snapshot — see
  §2.1.2).
- `session_search`: a local SQLite/FTS index of *raw* past transcripts, **not**
  injected; the agent must query for it on demand.

The split is deliberate: MEMORY.md is the always-on, hand-curated "what I should
never re-learn"; `session_search` is the searchable archive of "what happened
last time" you pay for only when you ask.

### 2.1 The `MEMORY.md` store

#### 2.1.1 Data model

- A single UTF-8 file (default `~/.mini-memory/MEMORY.md`).
- The file is a list of **entries** joined by a separator line:
  `ENTRY_SEPARATOR = "\n§\n"` (newline, section sign `§`, newline).
- On load: split on the separator, strip each entry, drop empties, and
  **deduplicate while preserving order** (`list(dict.fromkeys(entries))`).
- Capacity is measured in **characters of the joined string**, bounded by
  `char_limit` (default `48_000`).

#### 2.1.2 Frozen-snapshot semantics (the key idea)

- `load_snapshot()` is called once at session start: it reads disk, renders a
  display string, and caches it.
- `render_snapshot()` (used by `system_prompt_block`) returns that **cached**
  string for the entire session.
- `add`/`replace`/`remove` during the session write to **disk** but do **not**
  change the cached snapshot.

Consequences, and *why*:

- The system prompt is **byte-stable for the whole session** → preserves the
  LLM provider's prefix cache (huge cost saver over a 250-step trial).
- Writes therefore take effect in the **next** session, not the current one.
- To keep the model oriented anyway, **every successful tool response echoes the
  full live `entries` list and usage** (§2.1.5). The model edits against that
  live list, not against the frozen prompt copy.

There is intentionally **no `read` action**: the model sees memory via the
snapshot in its prompt and via the entries list in every tool response.

#### 2.1.3 The three actions

```text
add(content)               -> append a new entry
replace(old_text, content) -> rewrite the unique entry containing old_text
remove(old_text)           -> drop the unique entry containing old_text
```

Reference algorithm:

```python
def add(content):
    content = content.strip()
    if not content:                    return err("Content cannot be empty.")
    if bad := scan_invisible(content): return err(bad)
    entries = load()
    if content in entries:             return ok(entries, "Entry already exists (no duplicate added).")
    new = entries + [content]
    if chars(new) > char_limit:        return capacity_err(entries, chars(new), "adding", len(content))
    save(new);                         return ok(new, "Entry added.")

def replace(old_text, content):
    old_text, content = old_text.strip(), content.strip()
    if not old_text:                   return err("old_text cannot be empty.")
    if not content:                    return err("content cannot be empty. Use 'remove' to delete entries.")
    if bad := scan_invisible(content): return err(bad)
    entries = load()
    idx = unique_match(entries, old_text)
    if idx is None:                    return match_err(entries, old_text)
    new = entries.copy(); new[idx] = content
    if chars(new) > char_limit:        return capacity_err(entries, chars(new), "replacing", len(content))
    save(new);                         return ok(new, "Entry replaced.")

def remove(old_text):
    old_text = old_text.strip()
    if not old_text:                   return err("old_text cannot be empty.")
    entries = load()
    idx = unique_match(entries, old_text)
    if idx is None:                    return match_err(entries, old_text)
    save([e for i, e in enumerate(entries) if i != idx]); return ok(...)
```

#### 2.1.4 Supporting rules

- **`unique_match(entries, needle)`**: indices of entries containing `needle` as
  a substring. Return the index iff (a) exactly one entry matches, or (b)
  several match but all matching entries are byte-identical (return the first).
  Otherwise `None`.
- **`match_err`**: if 0 entries match → `"No entry contains <needle>."`; if >1
  distinct match → `"<n> entries match <needle>. Be more specific."` plus a
  `matches` list of 80-char previews.
- **`chars(entries)`** = `len(ENTRY_SEPARATOR.join(entries))`.
- **`pct(used, limit)`** = `min(100, round(100*used/limit))`.
- **Invisible/bidi unicode scan** (prompt-injection guard): reject content
  matching `[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]` with
  `"Blocked: content contains invisible unicode U+XXXX (possible injection)."`
- **Atomic write**: write to a tempfile in the same dir, `flush` + `fsync`,
  then `os.replace(tmp, path)`. Safe across crashes and concurrent readers;
  delete the tempfile on any error.

#### 2.1.5 Tool response shapes

Success:

```json
{ "success": true, "message": "Entry added.",
  "entries": ["...", "..."], "entry_count": 2,
  "usage": "13% — 6,400/48,000 chars" }
```

Capacity error (the model is expected to consolidate and retry):

```json
{ "success": false,
  "error": "Memory at 47,000/48,000 chars. Adding this entry (1200 chars) would push to 48,200, exceeding the limit. Replace or remove existing entries first.",
  "entries": [...], "entry_count": 12, "usage": "98% — 47,000/48,000 chars" }
```

#### 2.1.6 Rendered snapshot (what goes in the prompt)

If empty, the block is the empty string. Otherwise:

```text
══════════════════════════════════════════════
MEMORY (your persistent notes) [13% — 6,400/48,000 chars]
══════════════════════════════════════════════
<entry 1>
§
<entry 2>
```

(The bar is `═` × 46; entries are joined by `ENTRY_SEPARATOR`.)

#### 2.1.7 The `memory` tool schema (reproduce verbatim)

The tool's `description` is part of the method — it teaches *what* to save. Use
it as-is:

```json
{
  "type": "function",
  "function": {
    "name": "memory",
    "description": "Save durable engineering knowledge to MEMORY.md so it survives across trials and instances. Memory is injected into the system prompt at session start as a frozen snapshot — writes during this session take effect in the *next* session, but every tool response shows you the live entries list so you can plan consolidations.\n\nWHEN TO SAVE (be proactive, do not wait to be asked):\n- Project environment facts not obvious from the task input (language version, framework, test runner, build command, container/runtime quirks).\n- Build/test gotchas you verified ('pytest must run from repo root with PYTHONPATH=.', 'this repo's CI uses tox -e py311', 'tests/integration/* are flaky — retry').\n- Concrete bug-fix idioms specific to this codebase that you confirmed work.\n- Failed approaches you already ruled out so future trials don't repeat them.\n- Repo conventions (lint config, docstring style, type-checking rules, line width).\n\nPRIORITY: build/test infrastructure > stable bug-fix patterns > coding conventions. The most valuable entries prevent the next trial from re-discovering the same gotcha.\n\nDO NOT SAVE:\n- The current issue's text (already in the task input).\n- Raw logs, diffs, command output dumps, or stack traces.\n- Trial-local state (a path you cd'd to, a temp file you created).\n- Things easy to re-discover (file locations findable by ripgrep).\n\nACTIONS: 'add' (new entry), 'replace' (rewrite the entry uniquely identified by old_text), 'remove' (drop the entry uniquely identified by old_text). Keep entries compact and information-dense; consolidate via 'replace' when memory is over 80% full.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
        "content": {"type": "string", "description": "New entry text. Required for add/replace."},
        "old_text": {"type": "string", "description": "Unique substring of an existing entry. Required for replace/remove."}
      },
      "required": ["action"]
    }
  }
}
```

### 2.2 The `session_search` store

A local SQLite database with an FTS5 index over the **text of every past
session's messages**. Unlike MEMORY.md it is *not* injected; the agent recalls
from it on demand.

#### 2.2.1 Schema

```sql
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    started_at  REAL,
    ended_at    REAL,
    summary     TEXT,
    n_messages  INTEGER
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    session_id UNINDEXED,
    role       UNINDEXED,
    idx        UNINDEXED,
    content,                 -- the only indexed column
    tokenize='unicode61'
);
```

#### 2.2.2 Writing a session (`record_session`, idempotent)

Called from `on_session_end`. Idempotent on `session_id` so a session can be
re-flushed without duplicating rows:

```python
def record_session(session_id, messages, summary=""):
    if not session_id: return 0
    ended_at   = now()
    started_at = first_message_timestamp(messages) or ended_at   # extra.timestamp if present
    INSERT OR REPLACE INTO sessions VALUES (session_id, started_at, ended_at, summary, len(messages))
    DELETE FROM messages_fts WHERE session_id = session_id
    for idx, msg in enumerate(messages):
        content = extract_message_text(msg)        # §1.4
        if content:
            INSERT INTO messages_fts(session_id, role, idx, content)
                   VALUES (session_id, msg.role, idx, content)
```

The `summary` is supplied externally (see §2.2.4) and stored once per session.

#### 2.2.3 Searching (`search`, trial-level recall)

FTS ranks individual message hits, but results are **deduplicated by session**
so the model gets trial-level recall, not a bag of unrelated snippets.

```python
def search(query, limit=5):                      # limit clamped to [1, 20]
    query = sanitize_fts5_query(query)           # §2.2.5
    if not query: return []
    rows = SELECT m.session_id, s.summary, s.started_at, s.n_messages, m.role, m.idx,
                  snippet(messages_fts, 3, '<<', '>>', '...', 32) AS snippet
           FROM messages_fts m
           LEFT JOIN sessions s ON s.session_id = m.session_id
           WHERE messages_fts MATCH query
           ORDER BY bm25(messages_fts)
           LIMIT limit * 8                        # over-fetch, then dedup by session
    results, by_session = [], {}
    for row in rows:
        sid = row.session_id
        if sid not in by_session:
            if len(results) >= limit: continue
            by_session[sid] = {"session_id": sid,
                               "summary": truncate(row.summary, 2200),
                               "started_at": row.started_at,
                               "n_messages": row.n_messages,
                               "matches": []}
            results.append(by_session[sid])
        if len(by_session[sid]["matches"]) >= 3: continue   # ≤3 snippets/session
        by_session[sid]["matches"].append({
            "role": row.role, "idx": row.idx,
            "snippet": truncate(row.snippet, 350),
            "context": context(sid, row.idx, radius=1),      # neighbor messages ±1
        })
    for r in results: r["match_count"] = len(r["matches"])
    return results
```

- `snippet(messages_fts, 3, ...)`: column index **3** is `content`; markers
  `<<`/`>>`, ellipsis `...`, 32-token window.
- `context(sid, idx, radius=1)`: fetch messages `idx-1..idx+1` of that session,
  returning `{role, idx, content}` with content truncated to 350 chars.
- `truncate(text, n)`: if longer than `n`, cut to `n-14` chars + `"...[truncated]"`.
- If FTS raises `OperationalError` (bad query), return `[]`.

A `search` result therefore looks like:

```json
{ "success": true, "query": "ValidationError forms.py",
  "session_count": 1,
  "sessions": [{
    "session_id": "django__django-12345",
    "summary": "Task: ...\nCommands/tools: pytest ...\nFiles/symbols: forms.py, ...",
    "n_messages": 84, "match_count": 2,
    "matches": [{"role": "user", "idx": 31, "snippet": "...<<ValidationError>>...",
                 "context": [{"role":"assistant","idx":30,"content":"..."}, ...]}]
  }]}
```

#### 2.2.4 The extractive summary (`summarize_session`)

A cheap, synchronous, deterministic summary (no LLM) tailored to SWE-bench
recall. Built by scanning the transcript and concatenating present sections,
then truncating the whole to 2200 chars:

| Section | Source | Cap |
| --- | --- | --- |
| `Task: …` | first `user` message text, whitespace-compacted | 900 chars |
| `Commands/tools: …` | bash commands & non-bash tool calls from `extra.actions`, `; `-joined | 18 items, 180 chars each |
| `Files/symbols: …` | regex-mined file paths / `Mod.func()` / `A::B` refs | 30 refs |
| `Important output: …` | lines matching `traceback|error|failed|failure|assert|exception|pytest|ruff|mypy|go test|npm test` | 10 lines, 220 chars each |
| `Final: …` | `extra.exit_status` + `extra.submission` of the last message that has them | 500 chars |

This summary is stored in `sessions.summary` and shown first in each search
result so the model can judge relevance before reading snippets.

#### 2.2.5 FTS5 query sanitization (`sanitize_fts5_query`)

User/model queries must be made safe for FTS5 while preserving useful operators:

1. Temporarily extract `"quoted phrases"` and set them aside.
2. Replace FTS metacharacters `+{}()"^` with spaces.
3. Collapse `**` → `*`; drop leading `*` on tokens (FTS forbids prefix `*`).
4. Strip a leading/trailing boolean operator (`AND`/`OR`/`NOT`).
5. Per token: keep bare `AND/OR/NOT` and `term*` as-is; if a token contains any
   of `. _ - : /` (file paths, dotted names), wrap it in double quotes
   (doubling inner quotes) so FTS treats it as a phrase; otherwise keep bare.
6. Restore the preserved quoted phrases.

#### 2.2.6 The `session_search` tool schema (reproduce verbatim)

```json
{
  "type": "function",
  "function": {
    "name": "session_search",
    "description": "Full-text search over transcripts of *past* sessions stored locally in a SQLite FTS5 index. This is recall for previous coding trials; past sessions are NOT injected into the prompt, so you have to query for what you want.\n\nUSE THIS PROACTIVELY when the current repo, file, failing test, stack trace, or dependency looks like something a previous trial may have touched. At the start of an issue, if there is a clear cross-session signal (same repo, similar error, same test framework, or familiar file path), search before deep investigation so you can reuse prior lessons. Skip it when the task looks new or the signal is weak; noisy searches are not useful.\n\nReturns up to `limit` past trial/session results. Each result includes the session_id, a synchronous extractive summary, and up to three matching snippets with nearby transcript context. Use the summary to decide whether the past trial is relevant, then inspect the snippets for concrete commands, files, tests, and errors.\n\nQuery syntax is SQLite FTS5: bare words AND together; use `OR`, prefix `term*`, or quoted phrases `\"...\"` for precision. Keep queries short (1-4 keywords).",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "FTS5 query string."},
        "limit": {"type": "integer", "description": "Max past sessions to return (default 5, hard cap 20)."}
      },
      "required": ["query"]
    }
  }
}
```

### 2.3 Lifecycle wiring for Method A

| Hook | Behavior |
| --- | --- |
| `initialize(session_id)` | `MEMORY.load_snapshot()`; reset step/consolidation counters |
| `system_prompt_block()` | `MEMORY.render_snapshot()` (the frozen block, possibly empty) |
| `get_tool_schemas()` | `[memory_schema, session_search_schema]` |
| `tool_names` | `{"memory", "session_search"}` |
| `handle_tool_call("memory", args)` | dispatch to `add`/`replace`/`remove` by `args["action"]` |
| `handle_tool_call("session_search", args)` | `search(args["query"], limit=args.get("limit"))`; empty query → error dict |
| `on_session_end(messages)` | `record_session(session_id, messages, summary=summarize_session(messages))`; never raise |

`on_session_end` must **never break agent shutdown** — wrap indexing in a
try/except that swallows errors.

### 2.4 Prompt wording for Method A (reproduce)

System prompt (the `{{ memory_block }}` placeholder is where
`system_prompt_block()` is injected):

```text
You are a helpful assistant that can interact with a computer shell to solve programming tasks.

{{ memory_block }}

Alongside `bash` you have a `memory` tool that writes to a persistent
`MEMORY.md` shared across SWE-bench trials and instances. Use it proactively
for durable engineering knowledge: build/test commands, repo conventions,
verified gotchas, and failed approaches to avoid. Do not save the current
issue text, raw logs, diffs, or anything trivially re-discoverable.
You may also have `session_search`, which searches past trial transcripts.
Use it proactively when previous work in the same repo or files may help.
```

Per-task instruction reminder (abbreviated — the operative line):

```text
Provide one or more tool calls. Use `bash` for shell work, `session_search`
for past-trial recall, and `memory` for durable notes.
... Your response MUST include AT LEAST ONE available tool call. It can be
`bash`, `session_search`, or `memory`; do not add a dummy `bash` call solely
to satisfy this rule.
```

Tool-call error retry hint (`format_error_template`):

```text
Every response needs to use at least one available tool. Use `bash` for shell
work; `session_search` and `memory` can be used without an accompanying
`bash` call when they are the right next action. If you have completed your
assignment, consult the first message about how to submit your solution.
```

> **Prompt↔tools consistency rule (must check before every run).** The
> system/instance/error prompts must describe *exactly* the tools actually
> registered — no more, no less. If you disable `session_search`, remove it
> from all three prompts too; otherwise the model acts on capabilities it
> doesn't have and the experiment is invalid.

### 2.5 Optional: LLM consolidation (off by default)

An optional add-on that periodically runs a **memory-only LLM turn** to tidy
MEMORY.md. Both triggers are off by default (each costs an extra `model.query`).

- `on_session_end=true`: after a session, run one consolidation pass before
  shutdown so the trial's lessons are persisted.
- `every_n_steps=N` (N>0): during the run, after `N` model calls have elapsed
  **without** a successful MEMORY.md write, run one mid-run consolidation pass
  (a checkpoint, in case the session later times out).

The pass: build a tagged transcript (`[role] text` per message); if it exceeds
~1,000,000 chars, fall back to a tail-truncated transcript (keep most recent
turns up to a char budget). Send a prompt that (a) shows current memory, (b)
shows the trace, (c) instructs the model to make **at most `max_actions`** (default 3)
`memory` calls only — any `bash`/`session_search` in the response is ignored —
preferring `replace`/`remove` over `add`. Apply the resulting `memory` actions
to the store. Failures are swallowed.

---

## 3. Method B — Filesystem chain memory

A memory layer with **no extra tools**: the agent reads and searches memory with
plain `bash` (`rg`/`grep`/`sed`/`find`). Memory lives in a per-chain directory of
Markdown files. The host writes *raw evidence* at task end; a memory-only LLM
pass writes the *distilled* notes. The design borrows the "inject a short
read/write policy, not the whole memory" idea from filesystem-style agent memory.

Principles:

- One directory per chain (≈ one repo / one related task sequence).
- Full trajectory is **rule-written** (by the host), never authored by the model.
- The model authors only distilled artifacts: a per-case `summary.md`, the
  `INDEX.md` row fields for the current case, and a `repo.md` update block for
  the current case.
- No vector DB / embeddings / rerank. Retrieval = shell search at runtime.
- The model must **never** see official evaluation results, and must never edit
  another case's raw evidence.

### 3.1 Directory layout

```text
{memory_home}/
└── fs/
    └── chains/
        └── {chain_id}/
            ├── README.md          # usage policy for the runtime model
            ├── INDEX.md           # chain-level routing table (one row per case)
            ├── repo.md            # chain/repo-level accumulated notes
            └── cases/
                └── {step_index}-{instance_id}/
                    ├── task.md        # rule-written, read-only
                    ├── trajectory.md  # rule-written, read-only (can be huge)
                    ├── patch.diff     # rule-written, read-only
                    └── summary.md     # LLM-written distilled summary
```

`chain_dir = {memory_home}/fs/chains/{safe(chain_id)}`. The case folder name is
`{step_index}-{instance_id}` (or just the instance id if no step index).
`safe(x)` = replace runs of `[^A-Za-z0-9._-]` with `_`, strip leading/trailing
`._-`, default `"default"`.

### 3.2 File roles & write boundaries

| File | Writer | Notes |
| --- | --- | --- |
| `README.md`, `INDEX.md` skeleton, `repo.md` skeleton | host (created if missing) | |
| `cases/*/task.md` | host | first user message text |
| `cases/*/trajectory.md` | host | rendered transcript; **read-only**, may be very long |
| `cases/*/patch.diff` | host | final submission diff; **read-only** |
| `cases/*/summary.md` | memory-only LLM pass | distilled, structured, no eval results |
| `INDEX.md` row fields (summary/files/tests) for current case | memory-only LLM pass | host owns the skeleton row + path cell |
| `repo.md` current-case update block | memory-only LLM pass | appended/replaced per case; older notes preserved |

**Forbidden for the model**: editing any historical case's `task.md` /
`trajectory.md` / `patch.diff`, and writing any `resolved`/`pass`/`fail`/`gold`/
`score` / official evaluation label anywhere. Because Method B uses plain
`bash`, these boundaries are enforced by the prompt policy (§3.6), not by a tool
whitelist.

### 3.3 Task-end algorithm (`on_session_end`)

```python
def on_session_end(messages, model=None):
    if not enabled: return
    ensure_layout()                                   # README/INDEX/repo skeletons + cases/
    case_id  = f"{step_index}-{session_id}"           # or session_id if no step_index
    case_dir = chain_dir / "cases" / case_id
    write(case_dir/"task.md",       "# Task\n\n" + first_user_text(messages).strip() + "\n")
    write(case_dir/"trajectory.md", render_trajectory_markdown(messages))
    write(case_dir/"patch.diff",    final_submission(messages))
    upsert_index_row(step, instance=session_id, summary="", files_symbols="",
                     tests_errors="", path=f"cases/{case_id}/summary.md")   # skeleton row

    if model is not None:                             # memory-only distillation pass
        payload = parse_json(query_no_tools_or_query(model, summary_prompt(
                      task, trajectory, patch, index_md, repo_md, case_path)))
        if s := sanitize_summary(payload["summary_md"]):
            write(case_dir/"summary.md", s + "\n")
        if row := payload.get("index_row"):
            upsert_index_row(step, session_id, row["summary"], row["files_symbols"],
                             row["tests_errors"], path=f"cases/{case_id}/summary.md")
        if upd := payload.get("repo_updates", "").strip():
            upsert_repo_updates(chain_dir/"repo.md", case_path, upd)
    # all model-pass errors are swallowed; evidence files are still written
```

Helpers:

- `first_user_text`: text of the first `user` message (§1.4).
- `final_submission`: scan messages in reverse; return the first
  `msg.extra.submission` that is a string (else `""`).
- `render_trajectory_markdown`: `# Trajectory\n` then per message
  `## {idx}. {role}\n\n{extract_message_text(msg)}\n`, skipping empties. No
  global char cap — the model uses `rg`/`sed -n` to read slices, not `cat`.
- `parse_json`: strip a leading ```` ```json ```` / ```` ``` ```` fence if
  present, then `json.loads`; must be an object.
- `sanitize_summary`: defensively strip any `## Outcome …` section
  (regex `(?ims)^##+\s*Outcome\b.*?(?=^##+\s|\Z)`) in case the model leaks
  outcome info.
- `query_no_tools_or_query`: prefer `model.query_no_tools`, else `model.query`,
  with a single `{"role":"user","content": prompt}` message.

### 3.4 `INDEX.md` upsert

`INDEX.md` is the chain entry point: one short row per case. Upsert by matching
the **path cell** (so the skeleton row written before distillation is later
upgraded in place rather than duplicated).

```python
ROW = f"| {step} | {instance} | {summary} | {files_symbols} | {tests_errors} | {path} |"
# cells are escaped: replace "|" -> "\|", newlines -> space, then strip.
# If a line ends with `| {path} |`, replace that line; else insert the row
# immediately before the "## Repo-Level Notes" heading (trimming blank lines).
```

Skeleton:

```md
# Chain Memory Index

## Cases

| Step | Instance | Summary | Files / Symbols | Tests / Errors | Path |
| --- | --- | --- | --- | --- | --- |

## Repo-Level Notes

See `repo.md`.
```

### 3.5 `repo.md` upsert

`repo.md` accumulates chain/repo-level notes. The model returns only the
**current case's** bullets (`repo_updates`); the host appends or replaces that
case's block, preserving all other cases' notes.

```python
block = f"### {case_path}\n\n{updates}\n"
# If a block `### {case_path}` already exists, replace it in place
#   (regex: ^###\s+{case_path}\n\n.*?(?=^###\s+|^##\s+|\Z), multiline+dotall).
# Else append under a "## Case Updates" section (create it if missing,
#   inserting before the next "## " heading).
```

Skeleton:

```md
# Repo Notes

## Test Commands

## Repo Conventions

## Repeated Patterns

## Gotchas

## Useful Files
```

### 3.6 The runtime read/write policy (`system_prompt_block`, reproduce)

This block is injected into the system prompt. `{chain_dir}` is the absolute
path; `MEMORY_CHAIN_DIR` is shell-quoted. The exact path matters — tell the
model not to shorten it.

```text
<filesystem_memory>
You have filesystem memory for this experiment chain at:
  {chain_dir}

For shell commands, copy this exact assignment first:
  MEMORY_CHAIN_DIR='{chain_dir}'
Do not infer or shorten this path; the useful files are under this exact directory.

Use bash to inspect it when prior instances in this chain may help.

Read path:
1. Briefly inspect README.md and INDEX.md.
2. Search with rg when available; otherwise use grep/find/sed for repo file names, test names, error strings, symbols, or issue keywords.
3. Prefer repo.md and cases/*/summary.md for reusable knowledge.
4. trajectory.md can be very long. Read it only when a summary is relevant but insufficient.
5. For trajectory.md, search first, then read targeted line ranges with commands like `sed -n '<start>,<end>p' cases/<case>/trajectory.md`.
6. Do not cat or read a whole trajectory.md unless there is no narrower way to answer the current task.
7. Do not read more than needed; usually 1-3 summaries plus small trajectory excerpts are enough.

Write path:
1. During the task, do not modify memory files unless explicitly instructed by the memory policy.
2. Never modify task.md, trajectory.md, or patch.diff.
3. At task end, the system writes evidence files and may ask a memory-only pass to update summary.md / INDEX.md / repo.md.
</filesystem_memory>
```

`README.md` (written into the chain dir) restates the same read/write policy and
the `MEMORY_CHAIN_DIR=...; cd "$MEMORY_CHAIN_DIR"` convention for the model to
discover on disk.

### 3.7 The memory-only distillation prompt (reproduce)

Sent (no tools) at task end. Returns one JSON object.

```text
You are updating a filesystem memory for SWE-bench style coding tasks.
Return exactly one JSON object with keys: summary_md, index_row, repo_updates.

Rules:
- Do not include official evaluation results, resolved/pass/fail labels, scores, or outcome judgments.
- summary_md must follow the project summary template with sections: Task, Problem Signature, Investigation Path, Effective Change, Failed Attempts, Reusable Lessons.
- Every reusable lesson should cite evidence from trajectory.md, patch.diff, or a section in summary_md.
- index_row must contain summary, files_symbols, tests_errors.
- repo_updates should contain only new evidence-backed Markdown bullets for repo.md. Do not return the full repo.md; the system will append or replace this case's repo update block while preserving older notes.
- Use an empty string for repo_updates when this case has no durable repo-level lesson.
- Use this case path for evidence references: {case_path}

<task.md>
{task}
</task.md>

<trajectory.md>
{trajectory}
</trajectory.md>

<patch.diff>
{patch}
</patch.diff>

<INDEX.md>
{index}
</INDEX.md>

<repo.md>
{repo}
</repo.md>
```

`summary.md` target template (no `Outcome` section, no eval labels; every lesson
points to evidence):

```md
# {instance_id}

## Task
Short statement of the problem. Do not copy the full issue.

## Problem Signature
- Errors:
- Tests:
- Files:
- Symbols:
- Keywords:

## Investigation Path
Key localization steps; keep reusable files, commands, tests, error signals.

## Effective Change
The fix approach and files touched. Do not state whether official eval passed.

## Failed Attempts
Paths tried that were ineffective, misleading, or risky.

## Reusable Lessons
- Lesson: ...
  Evidence: trajectory.md / patch.diff / a section above
```

### 3.8 System-prompt wording for Method B (reproduce)

```text
You are a helpful assistant that can interact with a computer shell to solve programming tasks.

{{ memory_block }}

If filesystem memory is shown above, use bash to inspect it when earlier
instances in this chain may help. Prefer INDEX.md, repo.md, and case
summary.md files before reading targeted trajectory excerpts.
```

The only registered tool remains `bash`; the instance/error prompts mention
`bash` only (no `memory`/`session_search`).

---

## 4. Configuration reference

The manager is built from a YAML/dict config. Recognized keys (and the dataclass
fields behind them):

```yaml
memory:
  home: ~/.mini-memory          # base dir; providers/filesystem derive subdirs from it
  char_limit: 48000             # MEMORY.md capacity in chars
  builtin_enabled: true         # false → no MEMORY.md tool and no snapshot block
  sessions_enabled: true        # false → no session_search tool / store
  sessions_path: ~/.mini-memory/sessions.db   # optional override
  consolidation:                # §2.5, all off by default
    on_session_end: false
    every_n_steps: 0
    max_actions: 3
    summary_max_chars: 4000
  filesystem:                   # §3
    enabled: false
    chain_id: default           # home defaults to memory.home if unset
```

Method selection by config:

- **Method A (default)**: `builtin_enabled: true`, `sessions_enabled: true`,
  `filesystem.enabled: false`. Tools: `memory`, `session_search`.
- **Method B (filesystem only)**: `builtin_enabled: false`,
  `sessions_enabled: false`, `filesystem.enabled: true`. Tools: `bash` only.

For chained experiments, give **each chain its own `home`** (e.g.
`{root}/{chain_id}`) so chains never share `MEMORY.md` / `sessions.db` /
filesystem dirs; pass `chain_id` and `step_index` through `initialize(...)`.

On-disk layout when both are enabled under one `home`:

```text
{home}/
├── MEMORY.md          # Method A: distilled notes (frozen-snapshot source)
├── sessions.db        # Method A: SQLite + FTS5 transcript index
└── fs/chains/{chain_id}/...   # Method B: per-chain markdown memory (§3.1)
```

---

## 5. Reproduction checklist

Method A:

- [ ] `MEMORY.md` store: entries split on `\n§\n`, order-preserving dedup,
      `char_limit` on the joined string, atomic write, invisible-unicode scan.
- [ ] Frozen snapshot: `load_snapshot()` once at session start; prompt uses the
      cached render; writes go to disk but not the cache; tool responses echo
      the live `entries` + `usage`.
- [ ] `add`/`replace`/`remove` with unique-substring matching and capacity
      errors; no `read` action.
- [ ] `session_search`: SQLite WAL + FTS5 schema; idempotent `record_session`;
      `search` over-fetches `limit*8`, dedups by session, ≤3 snippets + ±1
      context, bm25 order; `sanitize_fts5_query`; extractive `summarize_session`.
- [ ] Both tool schemas registered with the model verbatim; prompts mention
      exactly these tools.
- [ ] `on_session_end` records the session and never raises.

Method B:

- [ ] Per-chain dir with `README.md` / `INDEX.md` / `repo.md` skeletons and
      `cases/{step_index}-{instance_id}/`.
- [ ] Host writes `task.md` / `trajectory.md` / `patch.diff` and a skeleton
      `INDEX.md` row at task end.
- [ ] Memory-only LLM pass (no tools) returns `{summary_md, index_row,
      repo_updates}`; host writes `summary.md`, upserts the `INDEX.md` row, and
      upserts the `repo.md` case block; summary sanitized of `Outcome`.
- [ ] Read/write policy injected into the system prompt; only `bash` registered.
- [ ] No official eval results reach memory; historical evidence files are never
      model-edited.

General:

- [ ] The five host hooks are wired; memory tool results are treated as
      transport-success.
- [ ] Each chain gets an isolated memory `home`; `chain_id`/`step_index` flow
      through `initialize`.

---

## 6. Design invariants (do not break these when reproducing)

1. **Frozen snapshot ⇒ stable prefix.** The injected memory block must not change
   mid-session, to keep the LLM prefix cache warm. Writes are deferred to the
   next session; the live entries list in tool responses is how the model stays
   current.
2. **Distilled vs. raw split.** `MEMORY.md` / `summary.md` / `repo.md` are
   small, curated, evidence-backed distillations. `session_search` /
   `trajectory.md` are the raw archive, queried on demand. Never inject raw
   transcripts into the prompt.
3. **Only index what the model saw.** Exclude off-band debug fields (e.g.
   `raw_output`) from indexing/summarization.
4. **No evaluation leakage.** Downstream sessions must never see
   resolved/pass/fail/score for prior cases.
5. **Memory must never break the agent.** `on_session_end` and any consolidation
   pass are best-effort: swallow their errors so a memory failure can't fail an
   otherwise-successful task.
6. **Prompts and registered tools must match exactly.** Disabling a component
   means removing it from every prompt that mentions it.
