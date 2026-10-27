"""Count PyTorch operations dispatched by a callable."""

from collections import Counter
from collections.abc import Callable

from torch.utils._python_dispatch import TorchDispatchMode

_VIEW_OPS = {
    "alias",
    "as_strided",
    "detach",
    "expand",
    "permute",
    "select",
    "slice",
    "split",
    "split_with_sizes",
    "squeeze",
    "t",
    "transpose",
    "unsqueeze",
    "view",
    "_unsafe_view",
}


def _op_name(func: object) -> str:
    schema = getattr(func, "_schema", None)
    if schema is not None:
        # aten::addmm.out and aten::addmm.default are one operation family here.
        return str(schema.name).replace("::", ".")
    return ".".join(str(func).split(".", maxsplit=2)[:2])


class _OperationCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[_op_name(func)] += 1
        return func(*args, **({} if kwargs is None else kwargs))


def count_ops(fn: Callable[[], object]) -> dict[str, object]:
    """Count aten operations dispatched while ``fn`` runs.

    This counts PyTorch dispatcher operations, not hardware kernels. It is the
    software-level unit to which the calibration's 12.67 microseconds per
    operation is applied. Some operations are classified as view-only because
    they only change tensor metadata; every other operation is ``compute``.
    Fused implementations may map one dispatched operation to a different
    number of hardware kernels.
    """
    mode = _OperationCounter()
    with mode:
        fn()
    total = sum(mode.counts.values())
    view = sum(
        count
        for name, count in mode.counts.items()
        if name.removeprefix("aten.") in _VIEW_OPS
    )
    return {
        "total": total,
        "view": view,
        "compute": total - view,
        "by_name": dict(sorted(mode.counts.items())),
    }
