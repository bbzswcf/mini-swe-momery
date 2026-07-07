"""Built-in bounded memory store backed by a single `MEMORY.md` file.

Design points (see `notes/hermes-memory-digest.md` §1):

- Single store: `MEMORY.md` only — no USER profile.
- Frozen snapshot semantics: `load_snapshot()` captures a rendered string at
  session start; `render_snapshot()` returns that cached string for the rest
  of the session. Mid-session writes hit disk via `_save()` but **do not**
  affect the cached snapshot — this preserves the LLM prefix cache.
- Three actions: `add` / `replace` / `remove`. No `read` action — the agent
  sees the snapshot via the system prompt and the live `entries` list inside
  every successful tool response.
- Capacity bounded by char count; over-limit writes return a structured
  error so the model can consolidate and retry.
- Entries separated on disk by `§` (section sign).
- Atomic writes (tempfile + `os.replace`) — safe against crashes and
  concurrent readers.
- Content is scanned for invisible / bidi unicode before being accepted.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ENTRY_SEPARATOR = "\n§\n"

_INVISIBLE_PATTERN = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]")


@dataclass
class BuiltinMemoryConfig:
    path: Path = field(default_factory=lambda: Path.home() / ".mini-memory" / "MEMORY.md")
    char_limit: int = 48_000


class BuiltinMemory:
    def __init__(self, config: BuiltinMemoryConfig | None = None) -> None:
        self.config = config or BuiltinMemoryConfig()
        self._snapshot: str = ""

    def load(self) -> list[str]:
        if not self.config.path.exists():
            return []
        raw = self.config.path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        entries = [e.strip() for e in raw.split(ENTRY_SEPARATOR) if e.strip()]
        return list(dict.fromkeys(entries))

    def load_snapshot(self) -> str:
        """Capture a frozen snapshot of the current on-disk state. Call once per session start."""
        self._snapshot = self._render(self.load())
        return self._snapshot

    def render_snapshot(self) -> str:
        """Return the cached snapshot. Empty if `load_snapshot()` was never called."""
        return self._snapshot

    def add(self, content: str) -> dict:
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        if err := self._scan(content):
            return {"success": False, "error": err}
        entries = self.load()
        if content in entries:
            return self._success(entries, message="Entry already exists (no duplicate added).")
        new_entries = [*entries, content]
        if (used := self._chars(new_entries)) > self.config.char_limit:
            return self._capacity_error(entries, used, action="adding", new_chars=len(content))
        self._save(new_entries)
        return self._success(new_entries, message="Entry added.")

    def replace(self, old_text: str, content: str) -> dict:
        old_text = old_text.strip()
        content = content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not content:
            return {"success": False, "error": "content cannot be empty. Use 'remove' to delete entries."}
        if err := self._scan(content):
            return {"success": False, "error": err}
        entries = self.load()
        if (idx := self._unique_match(entries, old_text)) is None:
            return self._match_error(entries, old_text)
        new_entries = entries.copy()
        new_entries[idx] = content
        if (used := self._chars(new_entries)) > self.config.char_limit:
            return self._capacity_error(entries, used, action="replacing", new_chars=len(content))
        self._save(new_entries)
        return self._success(new_entries, message="Entry replaced.")

    def remove(self, old_text: str) -> dict:
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        entries = self.load()
        if (idx := self._unique_match(entries, old_text)) is None:
            return self._match_error(entries, old_text)
        new_entries = [e for i, e in enumerate(entries) if i != idx]
        self._save(new_entries)
        return self._success(new_entries, message="Entry removed.")

    def _save(self, entries: list[str]) -> None:
        """Atomic write: tempfile → fsync → os.replace. Safe across crashes and concurrent readers."""
        path = self.config.path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ENTRY_SEPARATOR.join(entries) if entries else ""
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".mem_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _render(self, entries: list[str]) -> str:
        if not entries:
            return ""
        used, limit = self._chars(entries), self.config.char_limit
        bar = "═" * 46
        header = f"{bar}\nMEMORY (your persistent notes) [{self._pct(used, limit)}% — {used:,}/{limit:,} chars]\n{bar}"
        return header + "\n" + ENTRY_SEPARATOR.join(entries)

    def _success(self, entries: list[str], message: str) -> dict:
        used, limit = self._chars(entries), self.config.char_limit
        return {
            "success": True,
            "message": message,
            "entries": entries,
            "entry_count": len(entries),
            "usage": f"{self._pct(used, limit)}% — {used:,}/{limit:,} chars",
        }

    def _capacity_error(self, entries: list[str], projected: int, *, action: str, new_chars: int) -> dict:
        used, limit = self._chars(entries), self.config.char_limit
        return {
            "success": False,
            "error": (
                f"Memory at {used:,}/{limit:,} chars. {action.capitalize()} this entry "
                f"({new_chars} chars) would push to {projected:,}, exceeding the limit. "
                f"Replace or remove existing entries first."
            ),
            "entries": entries,
            "entry_count": len(entries),
            "usage": f"{self._pct(used, limit)}% — {used:,}/{limit:,} chars",
        }

    @staticmethod
    def _scan(content: str) -> str | None:
        if (m := _INVISIBLE_PATTERN.search(content)) is not None:
            return f"Blocked: content contains invisible unicode U+{ord(m.group()):04X} (possible injection)."
        return None

    @staticmethod
    def _unique_match(entries: list[str], needle: str) -> int | None:
        matches = [i for i, e in enumerate(entries) if needle in e]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1 and len({entries[i] for i in matches}) == 1:
            return matches[0]
        return None

    @staticmethod
    def _match_error(entries: list[str], needle: str) -> dict:
        n = sum(1 for e in entries if needle in e)
        if n == 0:
            return {"success": False, "error": f"No entry contains {needle!r}."}
        previews = [e[:80] + ("..." if len(e) > 80 else "") for e in entries if needle in e]
        return {"success": False, "error": f"{n} entries match {needle!r}. Be more specific.", "matches": previews}

    @staticmethod
    def _chars(entries: list[str]) -> int:
        return len(ENTRY_SEPARATOR.join(entries)) if entries else 0

    @staticmethod
    def _pct(used: int, limit: int) -> int:
        return min(100, round(100 * used / limit)) if limit > 0 else 0
