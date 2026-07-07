#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import os
import random
import re
import shlex
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable, TypeVar

import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live
from typer.models import OptionInfo

from minisweagent import Environment
from minisweagent.agents import get_agent_class
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c swebench.yaml -c agent.max_iterations=50[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

# Host env vars forwarded into the container only for whitelisted instances
# (see ``run.proxy_instances`` in the config). Covers HTTP(S)/Go ecosystems —
# both the standard upper-case names and their lower-case aliases that some
# tools still consult.
_DEFAULT_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "GOPROXY",
    "GONOSUMDB",
    "GONOPROXY",
)


DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
    "rebench": "nebius/SWE-rebench",
    "swe-bench-pro": "ScaleAI/SWE-bench_Pro",
}

# Local-file basenames for known subsets (used when resolving offline files).
_LOCAL_SUBSET_FILENAMES = {
    "swe-bench-pro": ["swe_bench_pro.json", "swe_bench_pro.jsonl"],
    "verified": ["swe_bench_verified.json", "swe_bench_verified.jsonl"],
}

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()
# Per-instance lock used when patch extraction touches container state. Each
# worker holds its own env, so a coarse process-wide lock is sufficient and
# avoids interleaved git diff invocations against the same container id.
_PATCH_LOCK = threading.Lock()
T = TypeVar("T")


def _typer_default(value: T | OptionInfo, fallback: T) -> T:
    return fallback if isinstance(value, OptionInfo) else value


def _default_output_path(subset: str) -> Path:
    name = Path(subset).stem if Path(subset).suffix else subset
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "swebench"
    return Path("results") / f"swebench_{name}_{time.strftime('%Y%m%d_%H%M%S')}"


def _make_progress_tracking_agent_class(base_class: type) -> type:
    """Wrap *any* agent class with batch-progress reporting on ``step``.

    Used to keep the live multi-instance UI working while still letting
    ``agent_class`` be configurable (e.g. ``memory`` for ``MemoryAgent``).
    """

    class ProgressTrackingAgent(base_class):  # type: ignore[misc, valid-type]
        def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
            super().__init__(*args, **kwargs)
            self.progress_manager: RunBatchProgressManager = progress_manager
            self.instance_id = instance_id

        def step(self):
            self.progress_manager.update_instance_status(
                self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})"
            )
            return super().step()

    return ProgressTrackingAgent


def _load_local_dataset_file(path: Path) -> list[dict]:
    """Load a single .json (list of instances) or .jsonl file."""
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def _find_local_dataset(subset: str) -> Path | None:
    """Locate a local dataset file for ``subset``. Resolution order:

    1. ``$SWEBENCH_DATA_DIR`` if set.
    2. Walk up from CWD looking for a ``data/`` directory.
    """
    candidates = _LOCAL_SUBSET_FILENAMES.get(subset, [])
    if not candidates:
        return None
    search_dirs: list[Path] = []
    if env_dir := os.getenv("SWEBENCH_DATA_DIR"):
        search_dirs.append(Path(env_dir))
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        search_dirs.append(parent / "data")
    for directory in search_dirs:
        for name in candidates:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_swebench_dataset(subset: str, split: str) -> list[dict]:
    """Load a SWE-bench dataset, preferring local files over HuggingFace.

    Resolution order:
      1. ``subset`` is a path to an existing ``.json``/``.jsonl`` file.
      2. ``$SWEBENCH_DATA_DIR/<known_filename>``.
      3. ``./data/<known_filename>`` walked up from CWD.
      4. Fall back to ``datasets.load_dataset`` against the HF identifier from
         ``DATASET_MAPPING`` (or ``subset`` itself).
    """
    direct = Path(subset)
    if direct.suffix in (".json", ".jsonl") and direct.is_file():
        logger.info(f"Loading local dataset file {direct}")
        return _load_local_dataset_file(direct)
    if local := _find_local_dataset(subset):
        logger.info(f"Loading local dataset file {local}")
        return _load_local_dataset_file(local)
    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split} from HuggingFace")
    return list(load_dataset(dataset_path, split=split))


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance.

    For SWE-bench-Pro instances the dataset ships a ``dockerhub_tag`` and the
    image lives under ``<SWEAP_DOCKERHUB_USERNAME>/sweap-images:<tag>`` (default
    username ``jefzda``). Other subsets keep the historical
    ``swebench/sweb.eval.x86_64.*`` naming.
    """
    image_name = instance.get("image_name", None) or instance.get("docker_image", None)
    if image_name:
        return image_name
    if dockerhub_tag := instance.get("dockerhub_tag"):
        username = os.environ.get("SWEAP_DOCKERHUB_USERNAME", "jefzda")
        return f"{username}/sweap-images:{dockerhub_tag}"
    iid = instance["instance_id"]
    id_docker_compatible = iid.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()


def _instance_matches_any(instance_id: str, patterns: list[str]) -> bool:
    """Return True iff any regex in ``patterns`` matches ``instance_id``."""
    return any(re.search(p, instance_id) for p in patterns)


def get_sb_environment(config: dict, instance: dict) -> Environment:
    # Build a *per-instance* env_config so concurrent workers don't fight over
    # ``image`` / ``forward_env`` mutations on the shared config dict.
    env_config = dict(config.get("environment", {}))
    run_config = config.get("run", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] in ["singularity", "contree"]:
        env_config["image"] = "docker://" + image_name

    # Only forward host proxy env vars for explicitly whitelisted instances
    # (e.g. Go modules that need a registry mirror). Forwarding to every
    # instance pollutes the test environment and can break offline tests.
    proxy_patterns = run_config.get("proxy_instances", []) or []
    if proxy_patterns and _instance_matches_any(instance["instance_id"], proxy_patterns):
        proxy_vars = run_config.get("proxy_env_vars") or list(_DEFAULT_PROXY_ENV_VARS)
        forward = list(env_config.get("forward_env", []))
        for var in proxy_vars:
            if var not in forward:
                forward.append(var)
        env_config["forward_env"] = forward
        logger.info(f"Forwarding proxy env to instance {instance['instance_id']}")
    if env_config["environment_class"] == "docker":
        _mount_filesystem_memory_home(config, env_config)

    env = get_environment(env_config)
    if startup_command := run_config.get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    # Snapshot HEAD inside the container *before* the agent touches anything.
    # SWE-bench-Pro images often carry pre-applied vendor patches (pinned deps,
    # entrypoint tweaks) on top of ``instance['base_commit']``; diffing against
    # the dataset base would surface those as a "Reversed patch" at evaluation
    # time. Capturing container HEAD gives us a clean diff base.
    head = env.execute({"command": "git rev-parse HEAD"})
    env._base_commit = head["output"].strip() if head.get("returncode") == 0 else None
    return env


# Files we never want to surface in a swe-bench patch: lockfiles get rewritten
# by ``npm/yarn/pnpm install`` during environment bring-up, and project meta
# files (``pyproject.toml``, ``.gitignore``) are routinely tweaked by the image
# itself. Keeping them out of the diff avoids spurious evaluation failures.
_PATCH_EXCLUDE_PATHS = (
    "pyproject.toml",
    ".gitignore",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "npm-shrinkwrap.json",
)
_MAX_PATCH_BYTES = 5 * 1024 * 1024
_MAX_PATCH_FILES = 500


def _get_patch_base_commit(env: Environment, instance: dict) -> str:
    """Prefer the HEAD captured at agent start; fall back to ``base_commit``."""
    return getattr(env, "_base_commit", None) or instance.get("base_commit") or "HEAD"


def _build_diff_command(base_commit: str) -> str:
    excludes = " ".join(f"':(exclude){path}'" for path in _PATCH_EXCLUDE_PATHS)
    return f"git diff --ignore-submodules=all {shlex.quote(base_commit)} -- . {excludes}"


def _is_valid_patch(patch: str) -> bool:
    if not patch.strip():
        return False
    if len(patch.encode("utf-8", errors="ignore")) > _MAX_PATCH_BYTES:
        logger.warning("Discarding extracted patch: exceeds 5 MiB size limit")
        return False
    file_count = patch.count("\ndiff --git ") + (1 if patch.startswith("diff --git ") else 0)
    if file_count > _MAX_PATCH_FILES:
        logger.warning(f"Discarding extracted patch: touches {file_count} files (>{_MAX_PATCH_FILES})")
        return False
    return True


def _collect_git_patch(env: Environment, instance: dict) -> str:
    """Run ``git diff`` inside the container against the recorded base commit.

    Stages untracked files <1 MiB with ``git add -N`` so they show up in the
    diff. Falls back to ``git diff HEAD`` when the image is a shallow clone and
    the base sha is missing.
    """
    with _PATCH_LOCK:
        # Stage tiny untracked files so the diff can include them. Anything
        # larger is almost certainly a build artifact / venv blob.
        env.execute(
            {
                "command": (
                    "git ls-files --others --exclude-standard -z "
                    "| while IFS= read -r -d '' f; do "
                    "  [ -f \"$f\" ] && [ \"$(wc -c <\"$f\")\" -lt 1048576 ] && git add -N -- \"$f\" >/dev/null 2>&1; "
                    "done; true"
                )
            }
        )
        base = _get_patch_base_commit(env, instance)
        result = env.execute({"command": _build_diff_command(base)})
        output = result.get("output", "") or ""
        if "fatal: bad object" in output or result.get("returncode", 0) != 0:
            logger.warning(f"git diff against {base!r} failed; falling back to HEAD")
            result = env.execute({"command": _build_diff_command("HEAD")})
            output = result.get("output", "") or ""
        return output


def _select_patch_result(agent_patch: str | None, collect_patch: Callable[[], str]) -> str:
    result = agent_patch or ""
    if _is_valid_patch(result):
        return result
    fallback = collect_patch()
    return fallback if _is_valid_patch(fallback) else result


def update_preds_file(output_path: Path, instance_id: str, model_name: str, result: str):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)
    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]
    # SWE-bench-Pro instances ship extra structured context that the agent
    # benefits from seeing verbatim. Wrap each block in xml-ish tags so the
    # template can keep using a single ``{{task}}`` slot.
    if requirements := instance.get("requirements"):
        task += "\n\n<requirements>\n" + requirements + "\n</requirements>"
    if interface := instance.get("interface"):
        task += "\n\n<interface>\n" + interface + "\n</interface>"

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    env = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)
        agent_kwargs = dict(config.get("agent", {}))
        agent_class = get_agent_class(agent_kwargs.pop("agent_class", "default"))
        agent = _make_progress_tracking_agent_class(agent_class)(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **agent_kwargs,
        )
        run_kwargs = {"session_id": instance_id}
        if instance.get("_chain_id") is not None:
            run_kwargs["chain_id"] = instance["_chain_id"]
        if instance.get("_step_index") is not None:
            run_kwargs["step_index"] = instance["_step_index"]
        info = agent.run(task, **run_kwargs)
        exit_status = info.get("exit_status")
        result = info.get("submission")
        if env is not None and not _is_valid_patch(result or ""):
            try:
                result = _select_patch_result(result, lambda: _collect_git_patch(env, instance))
            except Exception as e:
                logger.warning(f"Patch extraction fallback failed for {instance_id}: {e}")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


def load_chain_nodes(path: Path) -> dict[str, list[dict]]:
    chains: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if line.strip():
            node = json.loads(line)
            chains[node["chain_id"]].append(node)
    return {chain_id: sorted(nodes, key=lambda node: node["step_index"]) for chain_id, nodes in chains.items()}


def order_instances_by_chains(instances: list[dict], chain_nodes_path: Path) -> dict[str, list[dict]]:
    by_id = {instance["instance_id"]: instance for instance in instances}
    chains: dict[str, list[dict]] = {}
    missing: list[str] = []
    for chain_id, nodes in load_chain_nodes(chain_nodes_path).items():
        chain_instances = []
        for node in nodes:
            if node["instance_id"] not in by_id:
                missing.append(node["instance_id"])
                continue
            instance = dict(by_id[node["instance_id"]])
            instance["_chain_id"] = chain_id
            instance["_step_index"] = node["step_index"]
            chain_instances.append(instance)
        if chain_instances:
            chains[chain_id] = chain_instances
    if missing:
        raise ValueError(f"{len(missing)} chain instances are missing from the loaded dataset; first missing: {missing[0]}")
    return chains


def flatten_chain_instances(chains: dict[str, list[dict]]) -> list[dict]:
    return [instance for chain in chains.values() for instance in chain]


def chain_config(config: dict, chain_id: str, memory_root: Path) -> dict:
    memory_home = (memory_root / chain_id).expanduser().resolve()
    return recursive_merge(
        config,
        {
            "agent": {
                "memory": {
                    "home": str(memory_home),
                    "filesystem": {"chain_id": chain_id},
                }
            }
        },
    )


def _mount_filesystem_memory_home(config: dict, env_config: dict) -> None:
    memory_cfg = config.get("agent", {}).get("memory", {})
    filesystem_cfg = memory_cfg.get("filesystem", {})
    if not filesystem_cfg.get("enabled") or not memory_cfg.get("home"):
        return
    host_home = Path(str(memory_cfg["home"])).expanduser().resolve()
    host_home.mkdir(parents=True, exist_ok=True)
    mount = f"{host_home}:{host_home}:rw"
    run_args = list(env_config.get("run_args") or ["--rm"])
    if mount not in run_args:
        run_args.extend(["-v", mount])
    env_config["run_args"] = run_args


def process_chain(
    chain_id: str,
    instances: list[dict],
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    memory_root: Path,
) -> None:
    config = chain_config(config, chain_id, memory_root)
    logger.info(f"Running chain {chain_id} with {len(instances)} instances")
    for instance in instances:
        process_instance(instance, output_dir, config, progress_manager)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    chain_nodes: Path | None = typer.Option(None, "--chain-nodes", help="JSONL chain manifest; runs instances sequentially within each chain", rich_help_panel="Data selection"),
    chain_selection_only: bool = typer.Option(False, "--chain-selection-only", help="Use --chain-nodes only to select/order instances; run normal per-instance workers", rich_help_panel="Data selection"),
    chain_workers: int | None = typer.Option(None, "--chain-workers", help="Number of chains to run in parallel; defaults to number of chains", rich_help_panel="Basic"),
    chain_memory_root: Path | None = typer.Option(None, "--chain-memory-root", help="Root directory for per-chain memory homes", rich_help_panel="Basic"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    shuffle = _typer_default(shuffle, False)
    redo_existing = _typer_default(redo_existing, False)
    chain_nodes = _typer_default(chain_nodes, None)
    chain_selection_only = _typer_default(chain_selection_only, False)
    chain_workers = _typer_default(chain_workers, None)
    chain_memory_root = _typer_default(chain_memory_root, None)
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
    if chain_selection_only and chain_nodes is None:
        raise ValueError("--chain-selection-only requires --chain-nodes")
    chains = order_instances_by_chains(instances, chain_nodes) if chain_nodes is not None else {}
    if chains:
        chain_instance_ids = {instance["instance_id"] for chain in chains.values() for instance in chain}
        instances = flatten_chain_instances(chains)
        if chain_selection_only:
            chains = {}
            logger.info(f"Chain selection-only mode: {len(chain_instance_ids)} instances")
        else:
            logger.info(f"Chain mode: {len(chains)} chains, {len(chain_instance_ids)} instances")
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
        if chains:
            existing = set(existing_instances)
            chains = {
                chain_id: [instance for instance in chain if instance["instance_id"] not in existing]
                for chain_id, chain in chains.items()
            }
            chains = {chain_id: chain for chain_id, chain in chains.items() if chain}
    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    if chains and chain_workers is None:
        chain_workers = len(chains)
    memory_root = chain_memory_root or output_path / "chain_memory"

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=chain_workers or workers) as executor:
            if chains:
                futures = {
                    executor.submit(process_chain, chain_id, chain, output_path, config, progress_manager, memory_root): chain_id
                    for chain_id, chain in chains.items()
                }
            else:
                futures = {
                    executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
                        "instance_id"
                    ]
                    for instance in instances
                }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()
