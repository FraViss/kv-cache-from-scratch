from collections.abc import Callable

import pytest
import torch

from kvcache import DynamicKVCache, GPT2Config, KVCache, StaticKVCache


@pytest.fixture(params=["dynamic", "static"])
def cache_factory(request: pytest.FixtureRequest) -> Callable[[GPT2Config, int], KVCache]:
    """Build an empty cache for a config and a batch size, once per cache implementation.

    The static cache is float64 with the default capacity of config.n_positions.
    """
    if request.param == "dynamic":
        return lambda config, batch_size: DynamicKVCache(config.n_layer)
    return lambda config, batch_size: StaticKVCache.from_config(
        config, batch_size, dtype=torch.float64
    )
