from collections.abc import Callable

import pytest
import torch

from kvcache import CapturableKVCache, DynamicKVCache, GPT2Config, KVCache, StaticKVCache


@pytest.fixture(params=["dynamic", "static", "capturable"])
def cache_factory(request: pytest.FixtureRequest) -> Callable[[GPT2Config, int], KVCache]:
    """Build an empty cache for a config and a batch size, once per cache implementation.

    The static and capturable caches are float64 on the CPU, with the default capacity
    of config.n_positions. The capturable cache runs in eager mode here.
    """
    if request.param == "dynamic":
        return lambda config, batch_size: DynamicKVCache(config.n_layer)
    cache_class = StaticKVCache if request.param == "static" else CapturableKVCache
    return lambda config, batch_size: cache_class.from_config(
        config, batch_size, dtype=torch.float64
    )
