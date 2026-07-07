"""Unit tests for the SQLite + FTS5-backed `SessionStore`.

Focus on the contract that `MemoryManager` and the `session_search` tool depend
on: idempotent per-session writes, structured-content extraction, FTS5 ranking,
and graceful behavior on degenerate inputs.
"""

from __future__ import annotations

from minisweagent.memory.session_store import SessionStore, extract_message_text, summarize_session


def test_record_then_search_round_trip_returns_one_hit_per_matching_message(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.record_session(
        "trial-1",
        [
            {"role": "user", "content": "How do I run pytest with PYTHONPATH set?"},
            {"role": "assistant", "content": "Use PYTHONPATH=src pytest -xvs"},
            {"role": "user", "content": "thanks!"},
        ],
    )
    sessions = store.search("pytest")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "trial-1"
    assert {h["role"] for h in sessions[0]["matches"]} == {"user", "assistant"}
    assert all("<<" in h["snippet"] and ">>" in h["snippet"] for h in sessions[0]["matches"])
    assert sessions[0]["matches"][0]["context"]


def test_record_is_idempotent_per_session_id_and_replaces_old_messages(tmp_path):
    """Re-recording the same session must not pile up duplicate FTS rows."""
    store = SessionStore(tmp_path / "s.db")
    store.record_session("s", [{"role": "user", "content": "alpha keyword"}])
    store.record_session("s", [{"role": "user", "content": "beta keyword"}])
    assert store.search("alpha") == []
    assert len(store.search("beta")) == 1


def test_extracts_text_from_structured_list_content(tmp_path):
    """Multimodal-style `content: [{type, text}, ...]` blocks must still be indexed."""
    store = SessionStore(tmp_path / "s.db")
    store.record_session(
        "s",
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "ripgrep beats grep"}, {"type": "image_url"}],
            }
        ],
    )
    assert len(store.search("ripgrep")) == 1


def test_indexes_responses_api_outputs_and_tool_observations(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.record_session(
        "s",
        [
            {
                "object": "response",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "inspect Builder.apply"}]},
                    {"type": "function_call", "name": "bash", "arguments": '{"command": "pytest -q"}'},
                ],
            },
            {"type": "function_call_output", "output": "AssertionError: bad patch"},
        ],
    )
    assert len(store.search("Builder.apply")) == 1
    assert len(store.search("pytest")) == 1
    assert len(store.search("AssertionError")) == 1


def test_empty_query_or_session_id_is_a_noop(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    assert store.record_session("", [{"role": "user", "content": "x"}]) == 0
    store.record_session("s", [{"role": "user", "content": "alpha"}])
    assert store.search("   ") == []


def test_limit_caps_returned_hits_and_creates_parent_dirs(tmp_path):
    """Both the storage path and the limit param are robust to common edge cases."""
    store = SessionStore(tmp_path / "nested" / "deep" / "s.db")
    assert (tmp_path / "nested" / "deep" / "s.db").exists()
    for i in range(10):
        store.record_session(f"s-{i}", [{"role": "user", "content": f"keyword filler-{i}"}])
    assert len(store.search("keyword", limit=3)) == 3
    store.close()


def test_messages_with_no_extractable_text_are_skipped_but_session_row_kept(tmp_path):
    """Empty/non-text-content messages don't enter the FTS index, but the session metadata still exists."""
    store = SessionStore(tmp_path / "s.db")
    inserted = store.record_session(
        "s",
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": [{"type": "image_url"}]},
            {"role": "user", "content": "real text"},
        ],
    )
    assert inserted == 1
    hits = store.search("real")
    assert len(hits) == 1 and hits[0]["n_messages"] == 3


def test_system_prompt_messages_are_indexed(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    assert store.record_session(
        "s",
        [
            {"role": "system", "content": "MEMORY says use pytest from repo root"},
            {"role": "user", "content": "actual task mentions parser"},
        ],
    ) == 2
    assert len(store.search("MEMORY")) == 1
    assert len(store.search("parser")) == 1


def test_extract_message_text_does_not_duplicate_raw_output_already_in_content():
    text = extract_message_text(
        {
            "role": "user",
            "content": "<returncode>0</returncode>\n<output>\nunique tool output</output>",
            "extra": {"raw_output": "unique tool output", "returncode": 0},
        }
    )
    assert text.count("unique tool output") == 1


def test_extract_message_text_ignores_raw_output_not_visible_to_llm():
    """Untruncated bash output lives in extra.raw_output but is NOT sent to the LLM;
    the agent's observation_template head/tail-truncates what the model sees.
    The extractor must mirror what the LLM saw, not the off-band raw dump."""
    truncated_observation = (
        "<returncode>0</returncode>\n<output_head>\nstart of grep\n</output_head>\n"
        "<elided_chars>1234567 characters elided</elided_chars>\n"
        "<output_tail>\nend of grep\n</output_tail>"
    )
    text = extract_message_text(
        {
            "role": "user",
            "content": truncated_observation,
            "extra": {"raw_output": "MIDDLE_OF_GREP_DUMP_NEVER_SEEN_BY_LLM", "returncode": 0},
        }
    )
    assert "MIDDLE_OF_GREP_DUMP_NEVER_SEEN_BY_LLM" not in text
    assert "start of grep" in text
    assert "end of grep" in text


def test_record_session_does_not_index_off_band_raw_output(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.record_session(
        "s",
        [
            {
                "role": "user",
                "content": "<returncode>0</returncode>\n<output>\nshort visible chunk\n</output>",
                "extra": {"raw_output": "BIG_RAW_NEEDLE only in raw_output"},
            }
        ],
    )
    assert store.search("BIG_RAW_NEEDLE") == []
    assert len(store.search("visible")) == 1


def test_sanitizes_code_like_queries_instead_of_raising(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    store.record_session("s", [{"role": "user", "content": "failure in tests/test_widget.py::test_builder-case"}])
    assert len(store.search("tests/test_widget.py::test_builder-case")) == 1
    assert store.search(":") == []


def test_synchronous_summary_extracts_task_commands_and_failures():
    summary = summarize_session(
        [
            {"role": "user", "content": "Fix the parser bug"},
            {"role": "assistant", "content": "Need to inspect src/parser.py and Parser.parse()"},
            {
                "role": "assistant",
                "content": "",
                "extra": {"actions": [{"tool_name": "bash", "args": {"command": "pytest tests/test_parser.py"}}]},
            },
            {"role": "user", "content": "AssertionError: expected Token"},
            {"role": "exit", "content": "done", "extra": {"exit_status": "Submitted", "submission": "patch"}},
        ]
    )
    assert "Fix the parser bug" in summary
    assert "pytest tests/test_parser.py" in summary
    assert "src/parser.py" in summary
    assert "Parser.parse()" in summary
    assert "AssertionError" in summary
    assert "Submitted" in summary


def test_search_result_is_small_enough_for_observation_template(tmp_path):
    store = SessionStore(tmp_path / "s.db")
    noisy = "target " + "x" * 5000
    store.record_session(
        "s",
        [{"role": "user", "content": noisy} for _ in range(20)],
        summary="target " + "summary " * 1000,
    )
    assert len(str(store.search("target", limit=1))) < 9000
