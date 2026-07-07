"""Concrete `MemoryProvider` implementations.

Currently shipped:

- `HindsightProvider` (local_embedded mode) — long-term memory with a knowledge
  graph, backed by the ``hindsight`` package's local PostgreSQL daemon.
- `Mem0Provider` (OSS local) — server-side LLM fact extraction + semantic search
  via ``mem0``'s open-source `Memory` API (chroma-backed by default).

Each provider is opt-in via a separate optional dependency in ``pyproject.toml``;
heavy third-party imports happen lazily at first use, so importing this
subpackage does not require either backend to be installed.
"""

from minisweagent.memory.providers.hindsight import HindsightConfig, HindsightProvider
from minisweagent.memory.providers.mem0 import Mem0Config, Mem0Provider

__all__ = ["HindsightConfig", "HindsightProvider", "Mem0Config", "Mem0Provider"]
