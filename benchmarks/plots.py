"""Draw the benchmark figures from the committed JSON in benchmarks/results/.

    uv run --group bench python -m benchmarks.plots [--out docs/figures]

Every figure carries the device and the measuring commit taken from its JSON files.
Nothing is measured here.
"""

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from benchmarks.results_io import (  # noqa: E402
    DTYPES,
    RESULTS_DIR,
    calibration_values,
    commit,
    device_name,
    gpu_file,
    load,
    roofline_points,
    suite_records,
    time_breakdown,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "docs" / "figures"
DPI = 150
COLORS = {
    "uncached": "#7f7f7f",
    "dynamic": "#d62728",
    "static": "#1f77b4",
    "graph": "#2ca02c",
}
FIGURE_NAMES = (
    "decode_gpu.png",
    "decode_cpu.png",
    "time_breakdown.png",
    "batch_throughput.png",
    "memory.png",
    "capacity.png",
    "roofline.png",
)


def _footer(fig: Figure, payloads: Iterable[dict[str, Any]]) -> None:
    sources = sorted({f"{device_name(p)}, commit {commit(p)}" for p in payloads})
    fig.text(0.01, 0.005, "Measured on " + "; ".join(sources), fontsize=7, color="#555555")


def _save(fig: Figure, out: Path, name: str) -> Path:
    path = out / name
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _by(records: list[dict[str, Any]], key: str, variant: str) -> list[dict[str, Any]]:
    return sorted((r for r in records if r.get("variant") == variant), key=lambda r: r[key])


def plot_decode_gpu(results: Path, out: Path) -> Path:
    payloads = [load(results, gpu_file(dtype)) for dtype in DTYPES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, dtype, payload in zip(axes, DTYPES, payloads):
        records = suite_records(payload, "decode")
        for variant in ("uncached", "dynamic", "static", "graph"):
            rows = _by(records, "context_len", variant)
            ax.plot(
                [r["context_len"] for r in rows],
                [r["median_s"] * 1e3 for r in rows],
                marker="o",
                color=COLORS[variant],
                label=variant,
            )
        for variant, style in (("static", "--"), ("graph", "--")):
            rows = _by(records, "context_len", variant)
            ax.plot(
                [r["context_len"] for r in rows],
                [r["roofline_per_token_s"] * 1e3 for r in rows],
                linestyle=style,
                color=COLORS[variant],
                alpha=0.7,
                label=f"{variant} roofline",
            )
        rows = _by(records, "context_len", "graph")
        ax.plot(
            [r["context_len"] for r in rows],
            [r["graph_floor_plus_bandwidth_per_token_s"] * 1e3 for r in rows],
            linestyle=":",
            color=COLORS["graph"],
            label="graph floor + bandwidth",
        )
        ax.set_yscale("log")
        ax.set_title(f"Decode at B=1, {dtype}")
        ax.set_xlabel("context length T (tokens)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("time per token (ms, log scale)")
    axes[1].legend(fontsize=8)
    fig.suptitle("GPU decode: time per token against context length")
    _footer(fig, payloads)
    return _save(fig, out, "decode_gpu.png")


def plot_decode_cpu(results: Path, out: Path) -> Path:
    payload = load(results, "decode_cpu_float32.json")
    records = suite_records(payload, "decode")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for variant in ("uncached", "dynamic", "static"):
        rows = _by(records, "context_len", variant)
        xs = [r["context_len"] for r in rows]
        ax.plot(
            xs,
            [r["median_s"] * 1e3 for r in rows],
            marker="o",
            color=COLORS[variant],
            label=variant,
        )
        ax.plot(
            xs,
            [r["roofline_per_token_s"] * 1e3 for r in rows],
            linestyle="--",
            color=COLORS[variant],
            alpha=0.7,
            label=f"{variant} roofline",
        )
    ax.set_yscale("log")
    ax.set_title("CPU decode at B=1, float32")
    ax.set_xlabel("context length T (tokens)")
    ax.set_ylabel("time per token (ms, log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _footer(fig, [payload])
    return _save(fig, out, "decode_cpu.png")


def plot_time_breakdown(results: Path, out: Path) -> Path:
    payloads = [load(results, gpu_file(dtype)) for dtype in DTYPES]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels: list[str] = []
    segment_colors = {
        "CPU overhead (structural, eager)": "#ff7f0e",
        "rest": "#c7c7c7",
        "kernel floor (structural, graph)": "#9467bd",
        "weights / read bandwidth": "#1f77b4",
        "cache buffer / read bandwidth": "#17becf",
        "residual": "#d62728",
    }
    used: set[str] = set()
    for position, (dtype, kind) in enumerate(
        (d, k) for d in DTYPES for k in ("eager", "graph")
    ):
        data = time_breakdown(results, dtype)
        if kind == "eager":
            parts = [
                ("CPU overhead (structural, eager)", data["eager"]["cpu_overhead"]),
                ("rest", data["eager"]["rest"]),
            ]
            measured = data["eager"]["measured"]
            labels.append(f"eager static\n{dtype}")
        else:
            parts = [
                ("kernel floor (structural, graph)", data["graph"]["kernel_floor"]),
                ("weights / read bandwidth", data["graph"]["weights"]),
                ("cache buffer / read bandwidth", data["graph"]["cache"]),
                ("residual", data["graph"]["residual"]),
            ]
            measured = data["graph"]["measured"]
            labels.append(f"graph\n{dtype}")
        bottom_positive = 0.0
        bottom_negative = 0.0
        for name, value in parts:
            ms = value * 1e3
            if ms >= 0:
                ax.bar(position, ms, bottom=bottom_positive, color=segment_colors[name],
                       label=None if name in used else name, edgecolor="white")
                if ms > 0.15:
                    ax.text(position, bottom_positive + ms / 2, f"{ms:.2f}", ha="center",
                            va="center", fontsize=7)
                bottom_positive += ms
            else:
                ax.bar(position, ms, bottom=bottom_negative, color=segment_colors[name],
                       label=None if name in used else name, hatch="//", edgecolor="black")
                ax.text(position, bottom_negative + ms / 2, f"{ms:.2f}", ha="center",
                        va="center", fontsize=7)
                bottom_negative += ms
            used.add(name)
        ax.plot([position - 0.4, position + 0.4], [measured * 1e3] * 2, color="black", linewidth=1)
        ax.text(position, measured * 1e3 + 0.15, f"measured {measured * 1e3:.2f}", ha="center",
                fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("time per token (ms)")
    ax.set_title("Where a decode step goes at B=1, T=512 (negative parts hatched)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    _footer(fig, payloads + [calibration_values(results)["payload"]])
    return _save(fig, out, "time_breakdown.png")


def plot_batch_throughput(results: Path, out: Path) -> Path:
    payloads = [load(results, gpu_file(dtype)) for dtype in DTYPES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, dtype, payload in zip(axes, DTYPES, payloads):
        records = [r for r in suite_records(payload, "batch") if not r.get("out_of_memory")]
        for variant in ("dynamic", "static", "graph"):
            rows = _by(records, "batch_size", variant)
            ax.plot([r["batch_size"] for r in rows], [r["tokens_per_s"] for r in rows],
                    marker="o", color=COLORS[variant], label=variant)
        ax.set_xscale("log", base=2)
        ax.set_title(f"Decode throughput, T=512, {dtype}")
        ax.set_xlabel("batch size B (log2)")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("tokens per second")
    axes[0].legend()
    _footer(fig, payloads)
    return _save(fig, out, "batch_throughput.png")


def plot_memory(results: Path, out: Path) -> Path:
    payloads = [load(results, gpu_file(dtype)) for dtype in DTYPES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, dtype, payload in zip(axes, DTYPES, payloads):
        for record in suite_records(payload, "memory"):
            if record.get("out_of_memory") or record["batch_size"] != 8:
                continue
            variant = record["variant"]
            tokens = [s["generated_tokens"] for s in record["snapshots"]]
            ax.plot(tokens, [s["memory_reserved"] / 2**20 for s in record["snapshots"]],
                    color=COLORS[variant], label=f"{variant} reserved")
            ax.plot(tokens, [s["memory_allocated"] / 2**20 for s in record["snapshots"]],
                    color=COLORS[variant], linestyle="--", label=f"{variant} allocated")
        ax.set_title(f"CUDA allocator during generation, B=8, {dtype}")
        ax.set_xlabel("generated tokens")
        ax.set_ylabel("memory (MiB)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    _footer(fig, payloads)
    return _save(fig, out, "memory.png")


def plot_capacity(results: Path, out: Path) -> Path:
    payloads = [load(results, gpu_file(dtype)) for dtype in DTYPES]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for dtype, payload, color in zip(DTYPES, payloads, ("#1f77b4", "#ff7f0e")):
        rows = sorted(suite_records(payload, "capacity"), key=lambda r: r["cache_capacity"])
        xs = [r["cache_capacity"] for r in rows]
        ax.plot(xs, [r["median_s"] * 1e3 for r in rows], marker="o", color=color,
                label=f"{dtype} measured")
        ax.plot(xs, [r["roofline_per_token_s"] * 1e3 for r in rows], linestyle="--", color=color,
                label=f"{dtype} roofline")
    ax.set_xscale("log", base=2)
    ax.set_title("Graph decode against cache capacity, B=8, T=64")
    ax.set_xlabel("cache capacity (tokens, log2)")
    ax.set_ylabel("time per token (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    _footer(fig, payloads)
    return _save(fig, out, "capacity.png")


def plot_roofline(results: Path, out: Path) -> Path:
    payloads = [load(results, gpu_file(dtype)) for dtype in DTYPES]
    calibration = calibration_values(results)
    bandwidth = calibration["cuda_read_GBs"] * 1e9
    points = roofline_points(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    intensities = [p["intensity"] for p in points]
    lo, hi = min(intensities) / 2, max(intensities) * 2
    grid = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    for dtype, style in (("float32", "-"), ("bfloat16", "-.")):
        peak = calibration["peak_TFLOPs"][dtype] * 1e12
        ax.plot(grid, [min(peak, x * bandwidth) / 1e12 for x in grid], color="black",
                linestyle=style, linewidth=1,
                label=f"roof {dtype}: {calibration['peak_TFLOPs'][dtype]:.1f} TFLOP/s, "
                      f"{calibration['cuda_read_GBs']:.0f} GB/s")
    markers = {"float32": "o", "bfloat16": "^"}
    kind_colors = {"decode static": COLORS["static"], "decode graph": COLORS["graph"],
                   "prefill": "#8c564b"}
    seen: set[tuple[str, str]] = set()
    for point in points:
        key = (point["kind"], point["dtype"])
        ax.scatter(point["intensity"], point["achieved_TFLOPs"], marker=markers[point["dtype"]],
                   color=kind_colors[point["kind"]], s=36,
                   label=None if key in seen else f"{point['kind']} ({point['dtype']})")
        seen.add(key)
        ax.annotate(point["label"], (point["intensity"], point["achieved_TFLOPs"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte, cost model)")
    ax.set_ylabel("achieved TFLOP/s (cost-model FLOPs / measured time)")
    ax.set_title("Roofline: decode (batch suite, T=512) and prefill (B=1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    _footer(fig, payloads + [calibration["payload"]])
    return _save(fig, out, "roofline.png")


def make_figures(results: Path = RESULTS_DIR, out: Path = FIGURES_DIR) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [
        plot_decode_gpu(results, out),
        plot_decode_cpu(results, out),
        plot_time_breakdown(results, out),
        plot_batch_throughput(results, out),
        plot_memory(results, out),
        plot_capacity(results, out),
        plot_roofline(results, out),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    for path in make_figures(args.results, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
