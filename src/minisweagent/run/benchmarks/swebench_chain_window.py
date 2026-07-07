"""Run SWE-bench-Pro chains where one agent solves a whole chain in one LLM window.

Plug-and-play sibling of ``swebench.py``: it reuses dataset loading, chain
ordering, docker env bring-up and patch-extraction helpers from there, but
replaces the per-instance worker with one ``ChainWindowAgent`` per chain.
The existing chain dispatcher / Memory paths are completely untouched.

Trajectories are saved per-instance (same filename layout as ``swebench.py``)
so downstream analysis tools keep working.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import typer
from rich.live import Live
from typer.models import OptionInfo

from minisweagent.agents.chain_window import ChainWindowAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    _PATCH_LOCK,
    _collect_git_patch,
    _is_valid_patch,
    _select_patch_result,
    filter_instances,
    flatten_chain_instances,
    get_sb_environment,
    load_swebench_dataset,
    order_instances_by_chains,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_pro_chain_window.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _typer_default(value, fallback):
    return fallback if isinstance(value, OptionInfo) else value


def _default_output_path(subset: str) -> Path:
    name = Path(subset).stem if Path(subset).suffix else subset
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "swebench"
    return Path("results") / f"swebench_{name}_chain_window_{time.strftime('%Y%m%d_%H%M%S')}"


def _render_task(instance: dict) -> str:
    task = instance["problem_statement"]
    if requirements := instance.get("requirements"):
        task += "\n\n<requirements>\n" + requirements + "\n</requirements>"
    if interface := instance.get("interface"):
        task += "\n\n<interface>\n" + interface + "\n</interface>"
    return task


def _save_instance_traj(
    *,
    output_dir: Path,
    instance: dict,
    info: dict,
    messages: list[dict],
    agent: ChainWindowAgent,
    extra_info: dict | None = None,
) -> Path:
    """Write a per-instance ``traj.json`` compatible with existing analysis tools."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    traj_path = instance_dir / f"{instance_id}.traj.json"
    data = {
        "info": {
            "model_stats": {"instance_cost": agent.cost, "api_calls": agent.n_calls},
            "config": {
                "agent": agent.config.model_dump(mode="json"),
                "agent_type": f"{agent.__class__.__module__}.{agent.__class__.__name__}",
            },
            "exit_status": info.get("exit_status", ""),
            "submission": info.get("submission", ""),
            "chain_window": {
                "compression_log": list(agent._compression_log),
                "completed_instance_ids": list(agent._completed_instance_ids),
                "last_input_tokens": agent._last_input_tokens,
                "compression": {
                    "enabled": agent.compression.enabled,
                    "model_window": agent.compression.model_window,
                    "threshold": agent.compression.threshold,
                },
            },
            **(extra_info or {}),
        },
        "messages": messages,
        "trajectory_format": "mini-swe-agent-1.1",
        "instance_id": instance_id,
    }
    traj_path.write_text(json.dumps(data, indent=2))
    return traj_path


class _ProgressChainWindowAgent(ChainWindowAgent):
    """``ChainWindowAgent`` + a progress callback so the batch UI stays alive."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id_for_step: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._progress_manager = progress_manager
        self._instance_id_for_step = instance_id_for_step or {"id": ""}

    def step(self):
        self._progress_manager.update_instance_status(
            self._instance_id_for_step.get("id", ""),
            f"Step {self._task_n_calls + 1:3d} (${self._task_cost:.2f}, ctx≈{self._last_input_tokens})",
        )
        return super().step()


def process_chain(
    chain_id: str,
    instances: list[dict],
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    logger.info(f"Running chain {chain_id} with {len(instances)} instances (chain-window)")
    model = get_model(config=config.get("model", {}))
    instance_pointer: dict = {"id": instances[0]["instance_id"]}

    env = None
    agent: _ProgressChainWindowAgent | None = None
    started_ids: list[str] = []

    try:
        first = instances[0]
        progress_manager.on_instance_start(first["instance_id"])
        progress_manager.update_instance_status(first["instance_id"], "Pulling/starting environment")
        env = get_sb_environment(config, first)
        agent_kwargs = dict(config.get("agent", {}))
        agent_kwargs.pop("agent_class", None)
        agent = _ProgressChainWindowAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id_for_step=instance_pointer,
            **agent_kwargs,
        )
        started_ids.append(first["instance_id"])

        for index, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            instance_pointer["id"] = instance_id
            if index > 0:
                progress_manager.on_instance_start(instance_id)
                progress_manager.update_instance_status(instance_id, "Pulling/starting environment")
                started_ids.append(instance_id)
                if env is not None:
                    try:
                        env.cleanup()
                    except Exception:
                        logger.warning(f"Failed to cleanup env after {instances[index - 1]['instance_id']}", exc_info=True)
                env = get_sb_environment(config, instance)
                agent.env = env

            task = _render_task(instance)
            exit_status: str | None = None
            result: str = ""
            extra_info: dict[str, Any] = {}
            messages_snapshot: list[dict] = []

            try:
                info = agent._run_one(task, first=(index == 0))
                exit_status = info.get("exit_status")
                result = info.get("submission") or ""
                if env is not None and not _is_valid_patch(result):
                    with _PATCH_LOCK:
                        try:
                            result = _select_patch_result(result, lambda: _collect_git_patch(env, instance))
                        except Exception as e:
                            logger.warning(f"Patch extraction fallback failed for {instance_id}: {e}")
            except Exception as e:
                logger.error(f"Error processing instance {instance_id} (chain={chain_id}): {e}", exc_info=True)
                exit_status, result = type(e).__name__, ""
                extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
                info = {"exit_status": exit_status, "submission": ""}

            messages_snapshot = agent._extract_instance_messages()
            agent._completed_instance_ids.append(instance_id)
            _save_instance_traj(
                output_dir=output_dir,
                instance=instance,
                info={"exit_status": exit_status, "submission": result},
                messages=messages_snapshot,
                agent=agent,
                extra_info=extra_info,
            )
            update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
            progress_manager.on_instance_end(instance_id, exit_status)
            agent._seal_completed_task()
    except Exception as e:
        logger.error(f"Chain {chain_id} crashed: {e}", exc_info=True)
        for inst in instances:
            if inst["instance_id"] not in {i["instance_id"] for i in instances[: len(started_ids)]}:
                progress_manager.on_uncaught_exception(inst["instance_id"], e)
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                pass


# fmt: off
@app.command(help="Run SWE-bench-Pro chains where each chain shares one LLM context window.")
def main(
    subset: str = typer.Option("swe-bench-pro", "--subset", help="SWE-bench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("test", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class (e.g. 'litellm_response')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    chain_nodes: Path = typer.Option(..., "--chain-nodes", help="JSONL chain manifest; chains are required for this runner", rich_help_panel="Data selection"),
    chain_workers: int | None = typer.Option(None, "--chain-workers", help="Number of chains to run in parallel; defaults to number of chains", rich_help_panel="Basic"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Path to config files", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (docker / singularity / ...)", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    shuffle = _typer_default(shuffle, False)
    redo_existing = _typer_default(redo_existing, False)
    chain_workers = _typer_default(chain_workers, None)
    model = _typer_default(model, None)
    model_class = _typer_default(model_class, None)
    environment_class = _typer_default(environment_class, None)
    output = _typer_default(output, "")

    output_path = Path(output) if output else _default_output_path(subset)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    instances = load_swebench_dataset(subset, split)
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    chains = order_instances_by_chains(instances, chain_nodes)
    instances = flatten_chain_instances(chains)
    logger.info(f"Chain-window mode: {len(chains)} chains, {len(instances)} instances")

    if not redo_existing and (output_path / "preds.json").exists():
        # Chain-window context cannot be partially recovered: if a chain stopped
        # midway, the second half ran without the first half's accumulated
        # messages/summary, which would silently change the experiment. So we
        # resume chain-by-chain: keep fully-done chains, restart any partial
        # chain from scratch (clearing its preds + traj dirs).
        preds_path = output_path / "preds.json"
        preds = json.loads(preds_path.read_text())
        fully_done: list[str] = []
        partial: list[str] = []
        for cid, chain in list(chains.items()):
            ids = [i["instance_id"] for i in chain]
            done = sum(1 for x in ids if x in preds)
            if done == len(ids):
                fully_done.append(cid)
                chains.pop(cid)
            elif done > 0:
                partial.append(cid)
                for x in ids:
                    preds.pop(x, None)
                    instance_dir = output_path / x
                    if instance_dir.exists():
                        for p in instance_dir.glob("*"):
                            p.unlink()
                        instance_dir.rmdir()
        preds_path.write_text(json.dumps(preds, indent=2))
        logger.info(
            f"Resume: {len(fully_done)} chains fully done (skipped), "
            f"{len(partial)} chains partially done (cleared + re-running): {partial[:5]}"
            f"{'...' if len(partial) > 5 else ''}"
        )

    logger.info(f"Building chain-window agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    n_instances = sum(len(chain) for chain in chains.values())
    progress_manager = RunBatchProgressManager(n_instances, output_path / f"exit_statuses_{time.time()}.yaml")
    chain_workers = chain_workers or max(1, len(chains))

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=chain_workers) as executor:
            futures = {
                executor.submit(process_chain, chain_id, chain, output_path, deepcopy(config), progress_manager): chain_id
                for chain_id, chain in chains.items()
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Chain {futures[future]} failed: {e}", exc_info=True)
            except KeyboardInterrupt:
                logger.info("Cancelling pending chains. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()


if __name__ == "__main__":
    app()
