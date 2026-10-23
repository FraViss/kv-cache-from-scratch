import pytest
import torch

import kvcache


def test_package_version() -> None:
    assert kvcache.__version__ == "0.1.0"


def test_torch_float64_matmul() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], dtype=torch.float64)
    torch.testing.assert_close(left @ right, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_available() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device="cuda")
    torch.testing.assert_close(left @ right, expected)
