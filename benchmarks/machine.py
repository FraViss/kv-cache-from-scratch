"""What the benchmark ran on, in JSON-serializable form, and the numeric settings."""

import os
import platform
import subprocess
from datetime import datetime

import torch

_SMI_FIELDS = (
    "name",
    "driver_version",
    "power.limit",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.max.sm",
)


def set_benchmark_numerics() -> None:
    """Run float32 matmuls at full precision, with TF32 off.

    ``"highest"`` is the PyTorch setting that does not round float32 matmuls to
    TF32. Every benchmark script calls this before measuring.
    """
    torch.set_float32_matmul_precision("highest")


def nvidia_smi_query() -> dict[str, str | int | float | None] | None:
    """Read the first GPU from nvidia-smi, or None when the command is missing or fails.

    Numeric fields stay None when the driver reports them as unavailable. Nothing
    is raised: a machine without the tool still gets a machine record.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(_SMI_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(parts) != len(_SMI_FIELDS):
        return None
    parsed: dict[str, str | int | float | None] = {}
    for field, raw in zip(_SMI_FIELDS, parts, strict=True):
        if field in ("name", "driver_version"):
            parsed[field] = raw
            continue
        try:
            number = float(raw)
        except ValueError:
            parsed[field] = None
            continue
        parsed[field] = int(number) if number.is_integer() else number
    return parsed


def _git_porcelain(*, outside_results: bool = False) -> str:
    command = ["git", "status", "--porcelain", "--untracked-files=all"]
    if outside_results:
        command.extend(
            [
                "--",
                ".",
                ":(exclude)benchmarks/results",
                ":(exclude)benchmarks/results/**",
            ]
        )
    status = subprocess.run(command, capture_output=True, text=True, check=False)
    return status.stdout if status.returncode == 0 else ""


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Refuse benchmarks when code or documentation has uncommitted changes.

    Changes below ``benchmarks/results/`` are allowed because sequential runs
    replace several committed JSON files before one result commit. Everything
    else must be committed, unless ``allow_dirty`` is explicitly true.
    """
    if allow_dirty:
        return
    dirty = _git_porcelain(outside_results=True).strip()
    if dirty:
        raise RuntimeError(
            "benchmark code must be committed before measuring; uncommitted "
            f"changes outside benchmarks/results/:\n{dirty}\n"
            "Commit or restore them, or pass --allow-dirty explicitly."
        )


def _git_state() -> tuple[str | None, bool, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    sha = commit.stdout.strip() if commit.returncode == 0 else None
    return (
        sha,
        bool(_git_porcelain().strip()),
        bool(_git_porcelain(outside_results=True).strip()),
    )


def _gpu_info() -> dict[str, str | int | list[int]] | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": props.total_memory,
        "capability": [props.major, props.minor],
    }


def machine_info() -> dict[str, object]:
    """Describe the machine and the numeric settings. Every value is JSON-serializable.

    The timestamp is ISO 8601 with a timezone. ``dirty`` is true when
    ``git status --porcelain`` is non-empty. GPU fields are None when CUDA is
    unavailable or nvidia-smi cannot be queried.
    """
    commit, dirty, dirty_outside_results = _git_state()
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_info(),
        "nvidia_smi": nvidia_smi_query(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "git_commit": commit,
        "dirty": dirty,
        "dirty_outside_results": dirty_outside_results,
    }
