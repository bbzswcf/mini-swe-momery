"""Memory subsystem.

Provides:

- `BuiltinMemory`: bounded `MEMORY.md` store with `add`/`replace`/`remove`
  + frozen-snapshot rendering for system-prompt injection.
- `MemoryProvider` ABC + `MemoryManager` for pluggable external backends.
- Concrete providers in `minisweagent.memory.providers` (Hindsight local,
  Mem0 OSS local) — each behind its own optional dependency.

See `notes/hermes-memory-digest.md` for design rationale.
"""

from minisweagent.memory.builtin import BuiltinMemory, BuiltinMemoryConfig
from minisweagent.memory.consolidation import consolidate_memory
from minisweagent.memory.filesystem import FileSystemMemory, FileSystemMemoryConfig
from minisweagent.memory.manager import (
    MEMORY_TOOL_SCHEMA,
    SESSION_SEARCH_TOOL_SCHEMA,
    ConsolidationConfig,
    MemoryManager,
    MemoryManagerConfig,
)
from minisweagent.memory.provider import MemoryProvider
from minisweagent.memory.session_store import SessionStore

__all__ = [
    "MEMORY_TOOL_SCHEMA",
    "SESSION_SEARCH_TOOL_SCHEMA",
    "BuiltinMemory",
    "BuiltinMemoryConfig",
    "ConsolidationConfig",
    "FileSystemMemory",
    "FileSystemMemoryConfig",
    "MemoryManager",
    "MemoryManagerConfig",
    "MemoryProvider",
    "SessionStore",
    "consolidate_memory",
]
