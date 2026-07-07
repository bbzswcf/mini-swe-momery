# Memory Prompt / Tool Review

## Core Takeaways

- `MEMORY.md` frozen snapshot is the most reliable migrated mechanism: memory is loaded into the system prompt at session start, and writes during a session affect later sessions only.
- `requirements` and `interface` in SWE-bench-Pro are strong task-context variables. They may improve solve rate, but should be treated separately from memory gains.
- Memory experiments should avoid forcing every turn to include `bash`; `session_search` and `memory` need to be valid standalone next actions.
- `consolidation.on_session_end` is a complex intervention. Keep it separate from the simplest `MEMORY.md` experiment.
- Tool support is currently reliable for `litellm_model` / `litellm_response_model`. Other model backends should not be assumed to expose memory tools until their `extra_tools` and `allowed_tools` paths are implemented.

## Recommended Experiment Ladder

1. Baseline: no memory.
2. Chain-ordered runs with `MEMORY.md` plus `session_search`.
3. Add session-end consolidation.
4. Compare external providers such as Hindsight or Mem0.

## Default Minimal Config

- Fix the backend to `litellm_response` for the ModelHub Responses path.
- Use `MemoryAgent` with built-in `MEMORY.md` and local `session_search`.
- Set `sessions_enabled: true`.
- Set `consolidation.on_session_end: false`.
- Run nodes in `data/swe_bench_pro_chain_experiment_nodes.jsonl` order by `chain_id` and `step_index`.

## Isolation Rules

- Use one memory home per chain, or clear memory state at each chain start.
- Do not let `session_search` see sessions from unrelated chains when measuring chain-local memory.
- Keep the model backend fixed for memory comparisons, preferably `litellm_response` for the ModelHub Responses path.

## Prompt / Tool Risks To Track

- Very long `requirements` / `interface` blocks may dilute attention and should be ablated.
- `session_search` quality depends on transcript indexing; Responses `function_call_output` messages must be indexed if session recall is evaluated.
- Consolidation should be measured by solve rate, memory quality, and skipped/error rate, not assumed beneficial.
