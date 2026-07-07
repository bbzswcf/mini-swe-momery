"""SQLite + FTS5 store for past session transcripts.

Hermes-style session search (digest §2): every finished session's transcript
is written to a local sqlite db with a FTS5 index over message contents. The
``session_search`` tool lets the agent find past trials whose context resembles
the current task without paying per-session prompt cost — unlike MEMORY.md the
content is **not** injected into the system prompt, only retrieved on demand.

Schema:
- ``sessions(session_id PK, started_at, ended_at, summary, n_messages)`` — one
  row per session, optionally carries an externally-supplied ``summary``.
- ``messages_fts(session_id, role, idx, content)`` — FTS5 virtual table; one
  row per non-empty message. ``content`` is the only indexed column.

``record_session`` is idempotent on ``session_id`` (replaces existing rows),
so the same session can be re-flushed without duplicating FTS rows.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class SessionStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at REAL,
                ended_at REAL,
                summary TEXT,
                n_messages INTEGER
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                session_id UNINDEXED,
                role UNINDEXED,
                idx UNINDEXED,
                content,
                tokenize='unicode61'
            );
            """
        )
        self._conn.commit()

    def record_session(self, session_id: str, messages: list[dict], *, summary: str = "") -> int:
        """Replace any existing rows for ``session_id`` and (re-)index ``messages``.

        Returns the number of FTS rows inserted (= count of messages with
        non-empty extracted text).
        """
        if not session_id:
            return 0
        ended_at = time.time()
        started_at = self._extract_first_timestamp(messages) or ended_at
        inserted = 0
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "INSERT OR REPLACE INTO sessions(session_id, started_at, ended_at, summary, n_messages) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, started_at, ended_at, summary, len(messages)),
            )
            cur.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
            for idx, msg in enumerate(messages):
                content = extract_message_text(msg)
                if not content:
                    continue
                cur.execute(
                    "INSERT INTO messages_fts(session_id, role, idx, content) VALUES (?, ?, ?, ?)",
                    (session_id, str(msg.get("role", "")), idx, content),
                )
                inserted += 1
        self._conn.commit()
        return inserted

    def search(self, query: str, *, limit: int = 5) -> list[dict]:
        """Return one result per matching past session, with compact context.

        FTS still ranks individual message hits, but results are deduplicated
        by ``session_id`` so the model sees trial-level recall instead of a
        bag of unrelated snippets.
        """
        query = sanitize_fts5_query(query)
        if not query:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT m.session_id, s.summary, s.started_at, s.n_messages, m.role, m.idx,
                       snippet(messages_fts, 3, '<<', '>>', '...', 32) AS snippet
                FROM messages_fts m
                LEFT JOIN sessions s ON s.session_id = m.session_id
                WHERE messages_fts MATCH ?
                ORDER BY bm25(messages_fts)
                LIMIT ?
                """,
                (query, max(1, int(limit)) * 8),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[dict] = []
        by_session: dict[str, dict] = {}
        for row in rows:
            session_id = row[0]
            if session_id not in by_session:
                if len(results) >= max(1, int(limit)):
                    continue
                by_session[session_id] = {
                    "session_id": session_id,
                "summary": _truncate(row[1] or "", 2200),
                    "started_at": row[2],
                    "n_messages": row[3],
                    "matches": [],
                }
                results.append(by_session[session_id])
            if len(by_session[session_id]["matches"]) >= 3:
                continue
            idx = int(row[5])
            by_session[session_id]["matches"].append(
                {
                    "role": row[4],
                    "idx": idx,
                    "snippet": _truncate(row[6] or "", 350),
                    "context": self._context(session_id, idx),
                }
            )
        for result in results:
            result["match_count"] = len(result["matches"])
        return results

    def _context(self, session_id: str, idx: int, *, radius: int = 1) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT role, idx, content
            FROM messages_fts
            WHERE session_id = ? AND CAST(idx AS INTEGER) BETWEEN ? AND ?
            ORDER BY CAST(idx AS INTEGER)
            """,
            (session_id, idx - radius, idx + radius),
        ).fetchall()
        return [
            {"role": r[0], "idx": int(r[1]), "content": _truncate(r[2], 350)}
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _extract_first_timestamp(messages: list[dict]) -> float | None:
        for msg in messages:
            ts = msg.get("extra", {}).get("timestamp") if isinstance(msg.get("extra"), dict) else None
            if isinstance(ts, (int, float)):
                return float(ts)
        return None


def extract_message_text(msg: dict) -> str:
    """Flatten mini/Responses/chat messages into the text the LLM actually saw.

    Off-band metadata such as ``extra.raw_output`` (the untruncated bash dump
    kept only for replay/debug; never sent to the model) is intentionally
    skipped: indexing or consolidating it can blow past API content limits
    (e.g. ModelHub's 10MB cap) and pollutes recall with text the LLM never
    actually observed.
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


def summarize_session(messages: list[dict], *, max_chars: int = 2200) -> str:
    """Synchronous extractive summary tailored to SWE-bench trial recall."""
    chunks: list[str] = []
    if task := _first_role_text(messages, "user"):
        chunks.append("Task: " + _truncate(_compact(task), 900))
    commands = _commands(messages)
    if commands:
        chunks.append("Commands/tools: " + "; ".join(commands[:18]))
    files = _code_references(messages)
    if files:
        chunks.append("Files/symbols: " + ", ".join(files[:30]))
    signals = _failure_signals(messages)
    if signals:
        chunks.append("Important output: " + " | ".join(signals[:10]))
    if final := _final_status(messages):
        chunks.append("Final: " + final)
    return _truncate("\n".join(chunks), max_chars)


def sanitize_fts5_query(query: str) -> str:
    query = str(query or "").strip()
    if not query:
        return ""
    quoted: list[str] = []

    def preserve(match: re.Match) -> str:
        quoted.append(match.group(0))
        return f"\x00Q{len(quoted) - 1}\x00"

    query = re.sub(r'"[^"]*"', preserve, query)
    query = re.sub(r'[+{}()"^]', " ", query)
    query = re.sub(r"\*+", "*", query)
    query = re.sub(r"(^|\s)\*", r"\1", query)
    query = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", query.strip())
    query = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", query.strip())
    tokens = []
    for token in query.split():
        if token.startswith("\x00Q") or token.upper() in {"AND", "OR", "NOT"} or token.endswith("*"):
            tokens.append(token)
        elif any(c in token for c in "._-:/"):
            tokens.append('"' + token.replace('"', '""') + '"')
        else:
            tokens.append(token)
    query = " ".join(tokens)
    for idx, value in enumerate(quoted):
        query = query.replace(f"\x00Q{idx}\x00", value)
    return query.strip()


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


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 14] + "...[truncated]"


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _first_role_text(messages: list[dict], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role and (text := extract_message_text(msg).strip()):
            return text
    return ""


def _commands(messages: list[dict]) -> list[str]:
    commands: list[str] = []
    for msg in messages:
        for action in (msg.get("extra") or {}).get("actions") or []:
            args = action.get("args") or {}
            if action.get("tool_name", "bash") == "bash" and args.get("command"):
                commands.append(_truncate(_compact(args["command"]), 180))
            elif action.get("tool_name"):
                commands.append(_truncate(_json_text({"tool": action.get("tool_name"), "args": args}), 180))
    return commands


def _code_references(messages: list[dict]) -> list[str]:
    refs: list[str] = []
    pattern = re.compile(
        r"(?<![\w/.-])(?:[\w.-]+/)+(?:[\w.-]+)(?:::[\w.-]+)?|"
        r"\b[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|c|cc|cpp|h|hpp|yaml|yml|toml|json|md)\b|"
        r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\(\)|"
        r"\b[A-Za-z_][A-Za-z0-9_]*\(\)|"
        r"\b[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*\b"
    )
    for msg in messages:
        for ref in pattern.findall(extract_message_text(msg)):
            if ref not in refs:
                refs.append(ref)
            if len(refs) >= 30:
                return refs
    return refs


def _failure_signals(messages: list[dict]) -> list[str]:
    signals: list[str] = []
    pattern = re.compile(r"(traceback|error|failed|failure|assert|exception|pytest|ruff|mypy|go test|npm test)", re.I)
    for msg in messages:
        for line in extract_message_text(msg).splitlines():
            line = _compact(line)
            if line and pattern.search(line):
                signals.append(_truncate(line, 220))
                if len(signals) >= 10:
                    return signals
    return signals


def _final_status(messages: list[dict]) -> str:
    for msg in reversed(messages):
        extra = msg.get("extra") or {}
        if status := extra.get("exit_status"):
            submission = extra.get("submission", "")
            return _truncate(f"{status}; submission={submission}", 500)
    return ""
