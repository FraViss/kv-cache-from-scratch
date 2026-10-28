"""Feasibility checks for CUDA graph capture and replay on this machine.

They follow the procedure documented by PyTorch: warm up on a side stream, capture
with torch.cuda.graph, then replay. A replay runs the captured kernels on the same
memory addresses, so inputs go in by copy_ into static buffers and outputs are read
from static buffers.
"""

from collections.abc import Callable

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def capture(step: Callable[[], None], warmup: int = 3) -> torch.cuda.CUDAGraph:
    """Warm step up on a side stream, then capture it into a new graph.

    Warmup runs the step for real, so callers reset any state it changed before
    replaying. Capture itself records kernels without running them.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def test_capture_and_replay_with_static_buffers() -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.zeros(4, 16, dtype=torch.float64, device="cuda")
    W = torch.randn(16, 8, generator=generator, dtype=torch.float64, device="cuda")
    b = torch.randn(8, generator=generator, dtype=torch.float64, device="cuda")
    out = torch.empty(4, 8, dtype=torch.float64, device="cuda")

    def step() -> None:
        out.copy_(torch.relu(x @ W + b))

    graph = capture(step)
    for _ in range(5):
        new_x = torch.randn(4, 16, generator=generator, dtype=torch.float64, device="cuda")
        x.copy_(new_x)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.relu(new_x @ W + b), atol=1e-12, rtol=0)


def test_device_side_counter_advances_on_replay() -> None:
    buf = torch.zeros(1, 8, dtype=torch.float64, device="cuda")
    pos = torch.zeros(1, dtype=torch.long, device="cuda")
    val = torch.zeros(1, 1, dtype=torch.float64, device="cuda")

    def step() -> None:
        buf.index_copy_(1, pos, val)
        pos.add_(1)

    graph = capture(step)
    buf.zero_()
    pos.zero_()
    values = [3.0, -1.5, 7.25, 42.0]
    for value in values:
        val.fill_(value)
        graph.replay()
    torch.cuda.synchronize()

    expected = torch.zeros(1, 8, dtype=torch.float64)
    expected[0, :4] = torch.tensor(values, dtype=torch.float64)
    assert torch.equal(buf.cpu(), expected)
    assert pos.item() == 4


def test_python_int_index_is_frozen_at_capture() -> None:
    """A Python int used to index during capture is baked into the graph.

    The slice buf[:, start:start + 1] is resolved on the host when the graph is
    captured: the kernel records a fixed address. Changing start afterwards changes
    nothing on replay, and every replay writes at the captured position. That is why
    the write index of the cache has to become a tensor on the GPU, advanced by a
    kernel inside the graph, before a decode step can be captured.
    """
    buf = torch.zeros(1, 8, dtype=torch.float64, device="cuda")
    val = torch.zeros(1, 1, dtype=torch.float64, device="cuda")
    start = 0

    def step() -> None:
        buf[:, start : start + 1].copy_(val)

    graph = capture(step)
    buf.zero_()
    start = 3
    val.fill_(7.0)
    graph.replay()
    torch.cuda.synchronize()

    assert buf[0, 0].item() == 7.0
    assert torch.count_nonzero(buf).item() == 1
    assert start == 3  # the host value did change; the graph ignored it
