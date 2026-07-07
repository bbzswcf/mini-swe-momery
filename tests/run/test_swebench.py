import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from minisweagent import package_dir
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.run.benchmarks.swebench import (
    _DEFAULT_PROXY_ENV_VARS,
    _PATCH_EXCLUDE_PATHS,
    _build_diff_command,
    _instance_matches_any,
    _is_valid_patch,
    _select_patch_result,
    chain_config,
    filter_instances,
    flatten_chain_instances,
    get_sb_environment,
    get_swebench_docker_image_name,
    load_chain_nodes,
    load_swebench_dataset,
    main,
    order_instances_by_chains,
    process_instance,
    remove_from_preds_file,
    update_preds_file,
)


def _make_model_from_fixture(text_outputs: list[str], cost_per_call: float = 1.0, **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from trajectory fixture data (raw text outputs)."""

    def parse_command(text: str) -> list[dict]:
        match = re.search(r"```mswea_bash_command\s*\n(.*?)\n```", text, re.DOTALL)
        return [{"command": match.group(1)}] if match else []

    return DeterministicModel(
        outputs=[make_output(text, parse_command(text), cost=cost_per_call) for text in text_outputs],
        cost_per_call=cost_per_call,
        **kwargs,
    )


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_swebench_end_to_end(github_test_data, tmp_path, workers, container_executable):
    """Test the complete SWEBench flow using the _test subset with deterministic model"""

    model_responses = github_test_data["model_responses"]

    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        # Use side_effect to create a new model instance for each worker
        mock_get_model.side_effect = lambda **kwargs: _make_model_from_fixture(model_responses, cost_per_call=0.1)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=workers,
            filter_spec="swe-agent__test-repo-1",
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
        )

    traj_file_path = package_dir.parent.parent / "tests" / "test_data" / "github_issue.traj.json"
    trajectory = json.loads(traj_file_path.read_text())

    last_message = trajectory[-1]["content"]

    instance_id = "swe-agent__test-repo-1"
    expected_result = {
        instance_id: {
            "model_name_or_path": "deterministic",
            "instance_id": instance_id,
            "model_patch": last_message,
        }
    }

    with open(tmp_path / "preds.json") as f:
        actual_result = json.load(f)

    assert actual_result == expected_result

    traj_output_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    output_trajectory = json.loads(traj_output_file.read_text())
    assert output_trajectory["messages"][-1]["content"] == last_message


def test_get_image_name_with_existing_image_name():
    """Test get_image_name when image_name is already provided"""
    instance = {"image_name": "custom/image:tag", "instance_id": "test__repo__1"}
    assert get_swebench_docker_image_name(instance) == "custom/image:tag"


def test_get_image_name_without_image_name():
    """Test get_image_name when image_name needs to be constructed"""
    instance = {"instance_id": "swe-agent__test-repo__1"}
    expected = "docker.io/swebench/sweb.eval.x86_64.swe-agent_1776_test-repo_1776_1:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_none_image_name():
    """Test get_image_name when image_name is explicitly None"""
    instance = {"image_name": None, "instance_id": "django__django__4.0"}
    expected = "docker.io/swebench/sweb.eval.x86_64.django_1776_django_1776_4.0:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_with_complex_instance_id():
    """Test get_image_name with complex instance_id containing multiple double underscores"""
    instance = {"instance_id": "project__sub__module__version__1.2.3"}
    expected = "docker.io/swebench/sweb.eval.x86_64.project_1776_sub_1776_module_1776_version_1776_1.2.3:latest"
    assert get_swebench_docker_image_name(instance) == expected


def test_get_image_name_swe_bench_pro_default_username():
    """SWE-bench-Pro images use the dockerhub_tag with the default jefzda username."""
    instance = {"instance_id": "ignored", "dockerhub_tag": "django-1.0"}
    assert get_swebench_docker_image_name(instance) == "jefzda/sweap-images:django-1.0"


def test_get_image_name_swe_bench_pro_env_override(monkeypatch):
    """SWEAP_DOCKERHUB_USERNAME overrides the default user."""
    monkeypatch.setenv("SWEAP_DOCKERHUB_USERNAME", "acme")
    instance = {"instance_id": "ignored", "dockerhub_tag": "flask-2.1"}
    assert get_swebench_docker_image_name(instance) == "acme/sweap-images:flask-2.1"


def test_get_image_name_existing_image_beats_dockerhub_tag():
    """An explicit image_name always wins over dockerhub_tag."""
    instance = {"instance_id": "x", "dockerhub_tag": "ignored", "image_name": "custom/img:1"}
    assert get_swebench_docker_image_name(instance) == "custom/img:1"


@pytest.mark.parametrize(
    ("patch_text", "expected"),
    [
        ("", False),
        ("   \n", False),
        ("diff --git a/x b/x\n@@ -1 +1 @@\n-old\n+new\n", True),
        ("diff --git a/x b/x\n" + "x" * (5 * 1024 * 1024 + 10), False),
    ],
)
def test_is_valid_patch_basic(patch_text, expected):
    assert _is_valid_patch(patch_text) is expected


def test_is_valid_patch_rejects_too_many_files():
    pieces = ["diff --git a/f0 b/f0\n@@\n-a\n+b\n"]
    pieces += [f"diff --git a/f{i} b/f{i}\n@@\n-a\n+b\n" for i in range(1, 600)]
    assert _is_valid_patch("\n".join(pieces)) is False


def test_build_diff_command_excludes_lockfiles():
    cmd = _build_diff_command("abc123")
    assert cmd.startswith("git diff --ignore-submodules=all abc123 -- .")
    for path in _PATCH_EXCLUDE_PATHS:
        assert f"':(exclude){path}'" in cmd


def test_build_diff_command_quotes_funky_base():
    assert "'evil ref'" in _build_diff_command("evil ref")


def test_select_patch_result_keeps_agent_submission():
    agent_patch = "diff --git a/src/app.py b/src/app.py\n@@\n-old\n+new\n"

    assert _select_patch_result(agent_patch, lambda: "diff --git a/patch.txt b/patch.txt\n@@\n-old\n+dirty\n") == agent_patch


def test_select_patch_result_uses_fallback_for_empty_agent_submission():
    fallback_patch = "diff --git a/src/fallback.py b/src/fallback.py\n@@\n-old\n+new\n"

    assert _select_patch_result("", lambda: fallback_patch) == fallback_patch


def test_load_swebench_dataset_direct_json_path(tmp_path):
    payload = [{"instance_id": "a"}, {"instance_id": "b"}]
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(payload))
    assert load_swebench_dataset(str(path), split="ignored") == payload


def test_load_swebench_dataset_jsonl(tmp_path):
    payload = [{"instance_id": "a"}, {"instance_id": "b"}]
    path = tmp_path / "ds.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in payload))
    assert load_swebench_dataset(str(path), split="ignored") == payload


def test_load_swebench_dataset_env_dir(tmp_path, monkeypatch):
    payload = [{"instance_id": "from-env"}]
    (tmp_path / "swe_bench_pro.json").write_text(json.dumps(payload))
    monkeypatch.setenv("SWEBENCH_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert load_swebench_dataset("swe-bench-pro", split="test") == payload


@pytest.mark.parametrize(
    ("instance_id", "patterns", "expected"),
    [
        ("django__django__1", [], False),
        ("django__django__1", ["^django"], True),
        ("requests__requests__1", ["^django", "^go-"], False),
        ("acme__go-mod__1", ["^django", "go-"], True),
        ("acme__go-mod__1", ["go-mod"], True),
        ("acme__service__1", [".*service.*"], True),
    ],
)
def test_instance_matches_any(instance_id, patterns, expected):
    assert _instance_matches_any(instance_id, patterns) is expected


class _DummyEnv:
    """Stand-in env that records the kwargs it was constructed with."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def execute(self, action, cwd: str = "", *, timeout: int | None = None):
        # Pretend `git rev-parse HEAD` succeeds with a known sha.
        return {"output": "deadbeef\n", "returncode": 0, "exception_info": ""}


def _patch_env_factory(monkeypatch):
    """Replace get_environment with a factory that returns ``_DummyEnv``."""
    monkeypatch.setattr("minisweagent.run.benchmarks.swebench.get_environment", lambda cfg: _DummyEnv(**cfg))


def test_get_sb_environment_no_proxy_by_default(monkeypatch):
    _patch_env_factory(monkeypatch)
    config = {"environment": {"forward_env": ["KEEP"]}, "run": {}}
    get_sb_environment(config, {"instance_id": "django__django__1", "image_name": "x:1"})
    assert _DummyEnv.last_kwargs["forward_env"] == ["KEEP"]


def test_get_sb_environment_proxy_whitelist_match(monkeypatch):
    _patch_env_factory(monkeypatch)
    config = {
        "environment": {"forward_env": ["KEEP"]},
        "run": {"proxy_instances": ["^acme__go-"]},
    }
    get_sb_environment(config, {"instance_id": "acme__go-mod__1", "image_name": "x:1"})
    forwarded = _DummyEnv.last_kwargs["forward_env"]
    assert forwarded[0] == "KEEP"
    for var in _DEFAULT_PROXY_ENV_VARS:
        assert var in forwarded


def test_get_sb_environment_proxy_whitelist_skip(monkeypatch):
    _patch_env_factory(monkeypatch)
    config = {
        "environment": {"forward_env": ["KEEP"]},
        "run": {"proxy_instances": ["^acme__go-"]},
    }
    get_sb_environment(config, {"instance_id": "django__django__1", "image_name": "x:1"})
    assert _DummyEnv.last_kwargs["forward_env"] == ["KEEP"]


def test_get_sb_environment_does_not_mutate_shared_config(monkeypatch):
    """Per-instance image / forward_env writes must not leak into the shared dict."""
    _patch_env_factory(monkeypatch)
    config = {
        "environment": {"forward_env": ["KEEP"]},
        "run": {"proxy_instances": [".*"]},
    }
    snapshot_env = json.loads(json.dumps(config["environment"]))
    get_sb_environment(config, {"instance_id": "x__y__1", "image_name": "img:1"})
    assert config["environment"] == snapshot_env


def test_get_sb_environment_custom_proxy_env_vars(monkeypatch):
    _patch_env_factory(monkeypatch)
    config = {
        "environment": {"forward_env": []},
        "run": {"proxy_instances": [".*"], "proxy_env_vars": ["MY_ONLY_VAR"]},
    }
    get_sb_environment(config, {"instance_id": "anything", "image_name": "x:1"})
    assert _DummyEnv.last_kwargs["forward_env"] == ["MY_ONLY_VAR"]


def test_load_swebench_dataset_walk_up(tmp_path, monkeypatch):
    payload = [{"instance_id": "from-walk-up"}]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "swe_bench_pro.json").write_text(json.dumps(payload))
    nested = tmp_path / "deep" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.delenv("SWEBENCH_DATA_DIR", raising=False)
    monkeypatch.chdir(nested)
    assert load_swebench_dataset("swe-bench-pro", split="test") == payload


def test_filter_instances_no_filters():
    """Test filter_instances with no filtering applied"""
    instances = [{"instance_id": "repo1__test1"}, {"instance_id": "repo2__test2"}, {"instance_id": "repo3__test3"}]
    result = filter_instances(instances, filter_spec="", slice_spec="")
    assert result == instances


def test_filter_instances_regex_filter():
    """Test filter_instances with regex filtering"""
    instances = [
        {"instance_id": "django__test1"},
        {"instance_id": "flask__test2"},
        {"instance_id": "django__test3"},
        {"instance_id": "requests__test4"},
    ]
    result = filter_instances(instances, filter_spec=r"django__.*", slice_spec="")
    expected = [{"instance_id": "django__test1"}, {"instance_id": "django__test3"}]
    assert result == expected


def test_filter_instances_slice_only():
    """Test filter_instances with slice specification"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(10)]
    result = filter_instances(instances, filter_spec="", slice_spec="2:5")
    expected = [{"instance_id": "repo2__test2"}, {"instance_id": "repo3__test3"}, {"instance_id": "repo4__test4"}]
    assert result == expected


def test_filter_instances_slice_start_only():
    """Test filter_instances with slice start only"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(5)]
    result = filter_instances(instances, filter_spec="", slice_spec="3:")
    expected = [{"instance_id": "repo3__test3"}, {"instance_id": "repo4__test4"}]
    assert result == expected


def test_filter_instances_slice_end_only():
    """Test filter_instances with slice end only"""
    instances = [{"instance_id": f"repo{i}__test{i}"} for i in range(5)]
    result = filter_instances(instances, filter_spec="", slice_spec=":2")
    expected = [{"instance_id": "repo0__test0"}, {"instance_id": "repo1__test1"}]
    assert result == expected


def test_filter_instances_filter_and_slice():
    """Test filter_instances with both filtering and slicing"""
    instances = [
        {"instance_id": "django__test1"},
        {"instance_id": "flask__test2"},
        {"instance_id": "django__test3"},
        {"instance_id": "django__test4"},
        {"instance_id": "requests__test5"},
    ]
    result = filter_instances(instances, filter_spec=r"django__.*", slice_spec="1:3")
    expected = [{"instance_id": "django__test3"}, {"instance_id": "django__test4"}]
    assert result == expected


def test_filter_instances_shuffle():
    """Test filter_instances with shuffle enabled produces deterministic results"""
    instances = [{"instance_id": f"repo{i:02d}__test{i}"} for i in range(10)]
    # Test that shuffle produces same result with same seed
    result1 = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=True)
    result2 = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=True)
    assert result1 == result2
    # Test that shuffled result is different from original order
    result_no_shuffle = filter_instances(instances.copy(), filter_spec="", slice_spec="", shuffle=False)
    assert result1 != result_no_shuffle


def test_filter_instances_empty_list():
    """Test filter_instances with empty input list"""
    result = filter_instances([], filter_spec=r".*", slice_spec="0:5", shuffle=True)
    assert result == []


def test_filter_instances_no_matches():
    """Test filter_instances when regex matches nothing"""
    instances = [{"instance_id": "django__test1"}, {"instance_id": "flask__test2"}]
    result = filter_instances(instances, filter_spec=r"nonexistent__.*", slice_spec="")
    assert result == []


def test_load_chain_nodes_orders_steps(tmp_path):
    path = tmp_path / "chains.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(node)
            for node in [
                {"chain_id": "c1", "instance_id": "b", "step_index": 2},
                {"chain_id": "c1", "instance_id": "a", "step_index": 1},
                {"chain_id": "c2", "instance_id": "x", "step_index": 1},
            ]
        )
    )
    chains = load_chain_nodes(path)
    assert [node["instance_id"] for node in chains["c1"]] == ["a", "b"]
    assert [node["instance_id"] for node in chains["c2"]] == ["x"]


def test_order_instances_by_chains_uses_manifest_order(tmp_path):
    path = tmp_path / "chains.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(node)
            for node in [
                {"chain_id": "c1", "instance_id": "b", "step_index": 2},
                {"chain_id": "c1", "instance_id": "a", "step_index": 1},
            ]
        )
    )
    chains = order_instances_by_chains([{"instance_id": "a"}, {"instance_id": "b"}], path)
    assert [instance["instance_id"] for instance in chains["c1"]] == ["a", "b"]
    assert [instance["_chain_id"] for instance in chains["c1"]] == ["c1", "c1"]
    assert [instance["_step_index"] for instance in chains["c1"]] == [1, 2]


def test_flatten_chain_instances_preserves_chain_order():
    chains = {"c1": [{"instance_id": "a"}], "c2": [{"instance_id": "b"}]}
    assert [i["instance_id"] for i in flatten_chain_instances(chains)] == ["a", "b"]


def test_chain_config_sets_per_chain_memory_home(tmp_path):
    cfg = {"agent": {"memory": {"home": "base", "char_limit": 123}}, "model": {"model_name": "m"}}
    result = chain_config(cfg, "chain-1", tmp_path / "memory")
    assert result["agent"]["memory"]["home"] == str(tmp_path / "memory" / "chain-1")
    assert result["agent"]["memory"]["char_limit"] == 123
    assert result["agent"]["memory"]["filesystem"]["chain_id"] == "chain-1"
    assert result["model"]["model_name"] == "m"


def test_chain_config_resolves_relative_memory_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = chain_config({"agent": {"memory": {"filesystem": {"enabled": True}}}}, "chain-1", Path("memory"))
    assert result["agent"]["memory"]["home"] == str((tmp_path / "memory" / "chain-1").resolve())


def test_get_sb_environment_mounts_filesystem_memory_home_for_docker(monkeypatch, tmp_path):
    _patch_env_factory(monkeypatch)
    config = {
        "environment": {"run_args": ["--rm", "--network", "none"]},
        "agent": {
            "memory": {
                "home": str(tmp_path / "memory" / "chain-1"),
                "filesystem": {"enabled": True},
            }
        },
    }

    get_sb_environment(config, {"instance_id": "repo__issue-1", "image_name": "x:1", "_chain_id": "chain-1"})

    run_args = _DummyEnv.last_kwargs["run_args"]
    assert run_args[:3] == ["--rm", "--network", "none"]
    assert "-v" in run_args
    assert f"{tmp_path / 'memory' / 'chain-1'}:{tmp_path / 'memory' / 'chain-1'}:rw" in run_args


def test_process_instance_forwards_chain_metadata_to_agent_run(tmp_path, monkeypatch):
    seen: dict = {}

    class DummyModelConfig(BaseModel):
        model_name: str = "dummy"

    class DummyModel:
        config = DummyModelConfig()

        def serialize(self):
            return {"info": {}}

    class DummyEnv:
        pass

    class DummyAgent:
        def __init__(self, model, env, **kwargs):
            self.model = model
            self.env = env

        def run(self, task, **kwargs):
            seen["run_kwargs"] = kwargs
            return {
                "exit_status": "submitted",
                "submission": "diff --git a/x.py b/x.py\n@@\n-old\n+new\n",
            }

        def save(self, path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))

    class DummyProgress:
        def on_instance_start(self, instance_id):
            pass

        def update_instance_status(self, instance_id, status):
            pass

        def on_instance_end(self, instance_id, exit_status):
            pass

    monkeypatch.setattr("minisweagent.run.benchmarks.swebench.get_model", lambda config: DummyModel())
    monkeypatch.setattr("minisweagent.run.benchmarks.swebench.get_sb_environment", lambda config, instance: DummyEnv())
    monkeypatch.setattr("minisweagent.run.benchmarks.swebench.get_agent_class", lambda spec: DummyAgent)

    process_instance(
        {
            "instance_id": "repo__issue-1",
            "problem_statement": "Fix it",
            "_chain_id": "chain-a",
            "_step_index": 7,
        },
        tmp_path,
        {"agent": {"agent_class": "dummy"}, "model": {}},
        DummyProgress(),
    )

    assert seen["run_kwargs"] == {"session_id": "repo__issue-1", "chain_id": "chain-a", "step_index": 7}


def test_update_preds_file_new_file(tmp_path):
    """Test update_preds_file when output file doesn't exist"""
    output_path = tmp_path / "preds.json"
    update_preds_file(output_path, "test__instance__1", "test_model", "test_result")

    assert output_path.exists()
    result = json.loads(output_path.read_text())
    expected = {
        "test__instance__1": {
            "model_name_or_path": "test_model",
            "instance_id": "test__instance__1",
            "model_patch": "test_result",
        }
    }
    assert result == expected


def test_update_preds_file_existing_file(tmp_path):
    """Test update_preds_file when output file already exists"""
    output_path = tmp_path / "preds.json"

    # Create initial file with one instance
    initial_data = {
        "existing__instance": {
            "model_name_or_path": "old_model",
            "instance_id": "existing__instance",
            "model_patch": "old_result",
        }
    }
    output_path.write_text(json.dumps(initial_data))

    # Add new instance
    update_preds_file(output_path, "new__instance", "new_model", "new_result")

    result = json.loads(output_path.read_text())
    expected = {
        "existing__instance": {
            "model_name_or_path": "old_model",
            "instance_id": "existing__instance",
            "model_patch": "old_result",
        },
        "new__instance": {
            "model_name_or_path": "new_model",
            "instance_id": "new__instance",
            "model_patch": "new_result",
        },
    }
    assert result == expected


def test_update_preds_file_overwrite_existing(tmp_path):
    """Test update_preds_file overwrites existing instance"""
    output_path = tmp_path / "preds.json"

    # Create initial file
    initial_data = {
        "test__instance": {
            "model_name_or_path": "old_model",
            "instance_id": "test__instance",
            "model_patch": "old_result",
        }
    }
    output_path.write_text(json.dumps(initial_data))

    # Update existing instance
    update_preds_file(output_path, "test__instance", "new_model", "new_result")

    result = json.loads(output_path.read_text())
    expected = {
        "test__instance": {
            "model_name_or_path": "new_model",
            "instance_id": "test__instance",
            "model_patch": "new_result",
        }
    }
    assert result == expected


def test_remove_from_preds_file_existing(tmp_path):
    """Test remove_from_preds_file removes existing instance"""
    output_path = tmp_path / "preds.json"

    # Create file with multiple instances
    initial_data = {
        "instance1": {"model_name_or_path": "model1", "instance_id": "instance1", "model_patch": "result1"},
        "instance2": {"model_name_or_path": "model2", "instance_id": "instance2", "model_patch": "result2"},
    }
    output_path.write_text(json.dumps(initial_data))

    # Remove one instance
    remove_from_preds_file(output_path, "instance1")

    result = json.loads(output_path.read_text())
    expected = {"instance2": {"model_name_or_path": "model2", "instance_id": "instance2", "model_patch": "result2"}}
    assert result == expected


def test_remove_from_preds_file_nonexistent_instance(tmp_path):
    """Test remove_from_preds_file with nonexistent instance"""
    output_path = tmp_path / "preds.json"

    initial_data = {"instance1": {"model_name_or_path": "model1", "instance_id": "instance1", "model_patch": "result1"}}
    output_path.write_text(json.dumps(initial_data))

    # Try to remove nonexistent instance
    remove_from_preds_file(output_path, "nonexistent")

    # File should be unchanged
    result = json.loads(output_path.read_text())
    assert result == initial_data


def test_remove_from_preds_file_no_file(tmp_path):
    """Test remove_from_preds_file when file doesn't exist"""
    output_path = tmp_path / "preds.json"

    # Should not raise an error
    remove_from_preds_file(output_path, "any_instance")

    # File should still not exist
    assert not output_path.exists()


@pytest.mark.slow
def test_redo_existing_false_skips_existing(github_test_data, tmp_path):
    """Test that redo_existing=False skips instances that already have results"""
    model_responses = github_test_data["model_responses"]

    # Create existing preds.json with one instance
    preds_file = tmp_path / "preds.json"
    existing_data = {
        "swe-agent__test-repo-1": {
            "model_name_or_path": "previous_model",
            "instance_id": "swe-agent__test-repo-1",
            "model_patch": "previous_result",
        }
    }
    preds_file.write_text(json.dumps(existing_data))

    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.side_effect = lambda **kwargs: _make_model_from_fixture(model_responses)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=1,
            filter_spec="swe-agent__test-repo-1",
            redo_existing=False,
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
        )

    # Should still have the original result
    result = json.loads(preds_file.read_text())
    assert result == existing_data


@pytest.mark.slow
def test_redo_existing_true_overwrites_existing(github_test_data, tmp_path, container_executable):
    """Test that redo_existing=True processes instances even if they already have results"""
    model_responses = github_test_data["model_responses"]

    # Create existing preds.json with one instance
    preds_file = tmp_path / "preds.json"
    existing_data = {
        "swe-agent__test-repo-1": {
            "model_name_or_path": "previous_model",
            "instance_id": "swe-agent__test-repo-1",
            "model_patch": "previous_result",
        }
    }
    preds_file.write_text(json.dumps(existing_data))

    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.side_effect = lambda **kwargs: _make_model_from_fixture(model_responses, cost_per_call=0.1)

        main(
            subset="_test",
            split="test",
            slice_spec="0:1",
            output=str(tmp_path),
            workers=1,
            filter_spec="swe-agent__test-repo-1",
            redo_existing=True,
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
        )

    # Should have new result from deterministic model
    traj_file_path = package_dir.parent.parent / "tests" / "test_data" / "github_issue.traj.json"
    trajectory = json.loads(traj_file_path.read_text())
    expected_result = trajectory[-1]["content"]

    result = json.loads(preds_file.read_text())
    assert result["swe-agent__test-repo-1"]["model_patch"] == expected_result
    assert result["swe-agent__test-repo-1"]["model_name_or_path"] == "deterministic"


class ExceptionModelConfig(BaseModel):
    model_name: str = "exception_model"


class ExceptionModel:
    """Test model that raises exceptions during processing."""

    def __init__(self, exception_type: type[Exception] = RuntimeError, exception_message: str = "Test exception"):
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.cost = 0.0
        self.n_calls = 0
        self.config = ExceptionModelConfig()

    def query(self, *args, **kwargs):
        self.n_calls += 1
        raise self.exception_type(self.exception_message)

    def format_message(self, **kwargs) -> dict:
        return dict(**kwargs)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return [self.format_message(role="user", content=str(o)) for o in outputs]

    def get_template_vars(self, **kwargs) -> dict:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self) -> dict:
        return {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_exception_handling_in_agent_run(tmp_path, workers, container_executable):
    """Test that exceptions during agent.run() are properly handled and recorded"""
    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(RuntimeError, "Agent processing failed")

        with patch("minisweagent.run.benchmarks.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=workers,
                filter_spec="swe-agent__test-repo-1",
                config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
                environment_class="docker",
            )

    # Check that prediction file contains exception information
    preds_file = tmp_path / "preds.json"
    assert preds_file.exists()

    result = json.loads(preds_file.read_text())
    instance_id = "swe-agent__test-repo-1"
    assert instance_id in result
    assert result[instance_id]["model_patch"] == ""
    assert result[instance_id]["model_name_or_path"] == "exception_model"

    # Check that trajectory file contains exception information
    traj_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    assert traj_file.exists()

    traj_data = json.loads(traj_file.read_text())
    assert traj_data["instance_id"] == instance_id
    assert traj_data["info"]["exit_status"] == "RuntimeError"
    assert traj_data["info"]["submission"] == ""
    assert traj_data["info"]["exception_str"] == "Agent processing failed"


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_different_exception_types(tmp_path, workers, container_executable):
    """Test that different exception types are properly recorded"""
    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(ValueError, "Invalid input provided")

        with patch("minisweagent.run.benchmarks.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=workers,
                filter_spec="swe-agent__test-repo-1",
                config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
                environment_class="docker",
            )

    # Check trajectory file for correct exception type
    instance_id = "swe-agent__test-repo-1"
    traj_file = tmp_path / instance_id / f"{instance_id}.traj.json"
    traj_data = json.loads(traj_file.read_text())

    assert traj_data["info"]["exit_status"] == "ValueError"
    assert traj_data["info"]["submission"] == ""
    assert traj_data["info"]["exception_str"] == "Invalid input provided"


@pytest.mark.slow
def test_exception_handling_with_progress_manager(tmp_path, container_executable):
    """Test that progress manager receives exception notifications in multithreaded mode"""
    with patch("minisweagent.run.benchmarks.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = ExceptionModel(ConnectionError, "Network timeout")

        with patch("minisweagent.run.benchmarks.swebench.RunBatchProgressManager") as mock_progress_class:
            mock_progress_manager = mock_progress_class.return_value
            mock_progress_manager.render_group = None  # For Live context manager

            main(
                subset="_test",
                split="test",
                slice_spec="0:1",
                output=str(tmp_path),
                workers=2,  # Use multithreaded to test progress manager
                filter_spec="swe-agent__test-repo-1",
                config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
                environment_class="docker",
            )

            # Verify progress manager methods were called
            mock_progress_manager.on_instance_start.assert_called_once_with("swe-agent__test-repo-1")
            mock_progress_manager.on_instance_end.assert_called_once_with("swe-agent__test-repo-1", "ConnectionError")

            # on_uncaught_exception should not be called since exceptions are handled properly
            mock_progress_manager.on_uncaught_exception.assert_not_called()
