import os
import threading

import pytest

from minisweagent.memory.builtin import BuiltinMemory, BuiltinMemoryConfig


@pytest.fixture
def mem(tmp_path):
    return BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=200))


def test_default_char_limit_is_48k():
    assert BuiltinMemoryConfig().char_limit == 48_000


def test_add_load_roundtrip_preserves_order(mem):
    assert mem.add("hello world")["success"]
    assert mem.add("second entry")["success"]
    assert mem.load() == ["hello world", "second entry"]
    assert BuiltinMemory(mem.config).load() == ["hello world", "second entry"]


def test_add_success_response_includes_entries_and_usage(mem):
    res = mem.add("first fact")
    assert res["success"] and res["entries"] == ["first fact"] and res["entry_count"] == 1
    assert res["message"] == "Entry added." and "10/200" in res["usage"]


def test_duplicate_add_returns_success_with_message_and_does_not_grow(mem):
    mem.add("hello")
    res = mem.add("hello")
    assert res["success"] and res["entry_count"] == 1
    assert "no duplicate" in res["message"].lower()
    assert mem.load() == ["hello"]


def test_capacity_error_blocks_write_and_returns_current_state(mem):
    big = "x" * 150
    mem.add(big)
    res = mem.add("y" * 100)
    assert not res["success"] and "exceed" in res["error"].lower()
    assert res["entries"] == [big] and res["entry_count"] == 1
    assert "/200 chars" in res["usage"]
    assert mem.load() == [big]


@pytest.mark.parametrize(
    ("setup", "old_text", "expected_substr"),
    [
        ([], "x", "no entry"),
        (["entry one foo", "entry two foo"], "foo", "be more specific"),
    ],
)
def test_replace_match_errors(mem, setup, old_text, expected_substr):
    for e in setup:
        mem.add(e)
    res = mem.replace(old_text, "bar")
    assert not res["success"] and expected_substr in res["error"].lower()


def test_replace_substring_match_rewrites_entry(mem):
    mem.add("user prefers dark mode in editors")
    mem.add("project uses pytest")
    res = mem.replace("dark mode", "user prefers light mode")
    assert res["success"] and res["entries"] == ["user prefers light mode", "project uses pytest"]
    assert mem.load() == ["user prefers light mode", "project uses pytest"]


def test_remove_substring_drops_only_matched_entry(mem):
    for e in ("alpha", "beta", "gamma"):
        mem.add(e)
    res = mem.remove("alp")
    assert res["success"] and res["entries"] == ["beta", "gamma"]
    assert mem.load() == ["beta", "gamma"]
    assert not mem.remove("alp")["success"]


def test_replace_with_empty_content_rejected(mem):
    mem.add("entry")
    assert not mem.replace("entry", "")["success"]


def test_remove_with_empty_old_text_rejected(mem):
    assert not mem.remove("")["success"]


# ---------------------------------------------------------------------------
# Frozen snapshot semantics
# ---------------------------------------------------------------------------


def test_render_snapshot_returns_empty_until_load_snapshot_called(mem):
    mem.add("entry")
    assert mem.render_snapshot() == ""
    mem.load_snapshot()
    snap = mem.render_snapshot()
    assert "entry" in snap and "MEMORY" in snap and "/200 chars" in snap and "§" not in snap[: snap.index("entry")]


def test_snapshot_is_frozen_against_mid_session_writes(mem):
    mem.add("first")
    mem.load_snapshot()  # session starts here
    mem.add("second")  # mid-session write — must not appear in cached snapshot
    snap = mem.render_snapshot()
    assert "first" in snap and "second" not in snap
    mem.load_snapshot()  # next session refreshes
    assert "second" in mem.render_snapshot()


def test_render_snapshot_empty_when_no_entries(mem):
    mem.load_snapshot()
    assert mem.render_snapshot() == ""


# ---------------------------------------------------------------------------
# Safety / robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", ["hello\u200bworld", "evil\u202estring", "x\ufeffy"])
def test_invisible_unicode_blocked_on_write(mem, payload):
    mem.add("legit")
    assert not mem.add(payload)["success"]
    assert not mem.replace("legit", payload)["success"]
    assert mem.load() == ["legit"]


def test_atomic_write_leaves_no_partial_files_on_disk(mem):
    mem.add("entry one")
    mem.add("entry two")
    siblings = list(mem.config.path.parent.iterdir())
    assert siblings == [mem.config.path]


def test_load_dedupes_on_read(tmp_path):
    """If the file gets duplicate entries (e.g. concurrent write race), load() dedupes."""
    path = tmp_path / "MEMORY.md"
    path.write_text("dup\n§\ndup\n§\nunique", encoding="utf-8")
    assert BuiltinMemory(BuiltinMemoryConfig(path=path)).load() == ["dup", "unique"]


def test_concurrent_writes_do_not_corrupt_file(tmp_path):
    """Many threads adding entries: file must always be parseable, no torn writes."""
    mem = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md", char_limit=10_000))

    def writer(i: int) -> None:
        mem.add(f"entry-{i:03d}")

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = mem.load()
    assert all(e.startswith("entry-") for e in final)
    assert len(final) == len(set(final))


def test_save_failure_cleans_up_temp_file(tmp_path, monkeypatch):
    mem = BuiltinMemory(BuiltinMemoryConfig(path=tmp_path / "MEMORY.md"))

    def boom(*_a, **_kw):
        raise RuntimeError("simulated rename failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError, match="simulated rename failure"):
        mem.add("anything")
    assert list(tmp_path.iterdir()) == []
