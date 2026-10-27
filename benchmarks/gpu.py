"""GPU preparation and telemetry shared by benchmark scripts."""

import time

import torch

from benchmarks.machine import nvidia_smi_query


def heat_up(device: str | torch.device, seconds: float = 3.0) -> None:
    """Raise CUDA clocks with repeated float32 4096x4096 matmuls.

    The loop synchronizes after every matmul so wall time tracks completed GPU
    work. CPU is intentionally a no-op.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda" or seconds <= 0:
        return
    left = torch.randn(4096, 4096, dtype=torch.float32, device=resolved)
    right = torch.randn_like(left)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        left @ right
        torch.cuda.synchronize()
    del left, right
    torch.cuda.empty_cache()


def nvidia_smi_snapshot() -> dict[str, str | int | float | None] | None:
    """Return the shared nvidia-smi query without duplicating its parsing."""
    return nvidia_smi_query()
