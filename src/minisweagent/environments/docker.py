import logging
import os
import platform
import shlex
import subprocess
import uuid
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class DockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = ["--rm"]
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images. Used as the fallback when
    ``api_timeout`` is not set."""
    api_timeout: int | None = None
    """Timeout in seconds for synchronous Docker daemon API calls
    (``docker run``, ``docker stop`` during cleanup). Distinct from
    ``timeout`` (per-command exec) and ``container_timeout`` (in-container
    sleep). Falls back to ``pull_timeout`` when unset; bump this for big
    SWE-bench-Pro images where pulling + starting can take many minutes."""
    interpreter: list[str] = ["bash", "-lc"]
    """Interpreter to use to execute commands. Default is ["bash", "-lc"].
    The actual command will be appended as argument to this. Override this to e.g., modify shell flags
    (e.g., to remove the `-l` flag to disable login shell) or to use python instead of bash to interpret commands.
    """
    container_entrypoint: str | None = None
    """Override the image's ENTRYPOINT. Some SWE-bench-Pro images ship a custom
    ENTRYPOINT that exits immediately when given a ``sleep`` command; setting
    this to e.g. ``/bin/bash`` reroutes the start command through bash so the
    container stays alive.
    """
    mem_limit: str | None = None
    """Memory limit forwarded to ``docker run`` via ``--memory``/``--memory-swap``
    (e.g. ``"16g"``). Guards against runaway patches that would otherwise OOM
    the host."""


def _normalize_docker_image_ref(image: str) -> str:
    """Strip ``docker://`` so a Singularity-style image ref also works with
    ``docker``/``podman``. ``oci://`` is treated the same way."""
    for prefix in ("docker://", "oci://"):
        if image.startswith(prefix):
            return image[len(prefix) :]
    return image


class DockerEnvironment:
    def __init__(
        self,
        *,
        config_class: type = DockerEnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_container()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        image = _normalize_docker_image_ref(self.config.image)
        cmd: list[str] = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
        ]
        if self.config.mem_limit:
            cmd += ["--memory", self.config.mem_limit, "--memory-swap", self.config.mem_limit]
        cmd += list(self.config.run_args)
        if self.config.container_entrypoint:
            # Take over a baked-in ENTRYPOINT so our ``sleep`` keeps the
            # container alive instead of being passed as the entrypoint's
            # arguments. Using ``bash -c`` lets the sleep duration string be
            # parsed exactly like the unconfigured case.
            cmd += [
                "--entrypoint",
                self.config.container_entrypoint,
                image,
                "-c",
                f"sleep {self.config.container_timeout}",
            ]
        else:
            cmd += [image, "sleep", self.config.container_timeout]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.api_timeout or self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, *self.config.interpreter, command])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout or self.config.timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            api_timeout = self.config.api_timeout or self.config.pull_timeout
            cmd = (
                f"(timeout {api_timeout} {self.config.executable} stop {self.container_id} "
                f"|| {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            )
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
