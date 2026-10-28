"""Write docs/results.md: every benchmark table, from the committed JSON.

    uv run python -m benchmarks.summary [--out docs/results.md] [--no-test-counts]

Each table names its source file and the commit that measured it. Two short sections
are not from benchmark JSON and say where they come from: the correctness numbers
recorded in the decision records, and the test counts collected by pytest.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.costs import kv_bytes_per_token, weight_bytes
from benchmarks.results_io import (
    DTYPES,
    GPT2_SMALL,
    ITEMSIZE,
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
RESULTS_MD = ROOT / "docs" / "results.md"
SECTIONS = (
    "Provenance",
    "Correctness",
    "Calibration",
    "Cost model constants",
    "Decode on the GPU",
    "Decode on the CPU",
    "Time breakdown",
    "Overhead",
    "Batch",
    "Capacity",
    "Prefill",
    "Memory",
    "Roofline points",
    "Graph isolation",
    "Shapes",
)


def ms(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds * 1e3:.3f}"


def mib(n: float) -> str:
    return f"{n / 2**20:.1f}"


def table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def source(name: str, payload: dict[str, Any]) -> str:
    return f"Source: `benchmarks/results/{name}`, commit `{commit(payload)}`, {device_name(payload)}."


def section_provenance(results: Path) -> str:
    rows = []
    for path in sorted(results.glob("*.json")):
        payload = load(results, path.name)
        machine = payload["machine"]
        rows.append([
            f"`{path.name}`",
            f"`{commit(payload)}`",
            machine.get("dirty_outside_results"),
            machine["timestamp"][:19],
            f"{payload['duration_s']:.1f}" if "duration_s" in payload else "n/a",
        ])
    csvs = sorted(p.name for p in results.glob("*.csv"))
    text = table(["file", "commit", "dirty outside results", "measured at", "duration (s)"], rows)
    if csvs:
        text += "\n\nGPU clock logs: " + ", ".join(f"`benchmarks/results/{n}`" for n in csvs) + "."
    return text


def section_correctness(test_counts: dict[str, int] | None) -> str:
    lines = [
        "Not from benchmark JSON; recorded when each check was made.",
        "",
        "- Logits against Hugging Face `GPT2LMHeadModel`, float64, CPU, input (2, 64): "
        "maximum absolute difference 1.6e-13 (ADR-0004).",
        "- Cached against uncached logits, pretrained model, float64, 1000-token prompt plus "
        "24 decode steps: maximum absolute difference 8.3e-13 (`docs/status.md`, M2).",
        "- Greedy sequences: identical to Hugging Face `generate` for the uncached path, both "
        "eager caches and the CUDA graph decoder (integration tests in "
        "`tests/test_hf_equivalence.py`).",
    ]
    if test_counts is not None:
        lines.append(
            f"- Test counts collected by pytest: {test_counts['fast']} fast tests "
            f"(`uv run pytest`) and {test_counts['integration']} integration tests "
            "(`uv run pytest -m integration`)."
        )
    return "\n".join(lines)


def section_calibration(results: Path) -> str:
    payload = load(results, "calibration.json")
    rows = []
    for device, values in payload["results"].items():
        for name, value in values.items():
            if isinstance(value, dict):
                rows.append([device, name, f"{value['median']:.3f}", f"{value['p25']:.3f}",
                             f"{value['p75']:.3f}"])
            else:
                rows.append([device, name, f"{value:.3f}", "", ""])
    return source("calibration.json", payload) + "\n\n" + table(
        ["device", "quantity", "median", "p25", "p75"], rows
    )


def section_cost_model() -> str:
    rows = []
    for dtype in DTYPES:
        per_token = kv_bytes_per_token(GPT2_SMALL, ITEMSIZE[dtype])
        rows.append([
            dtype,
            f"{weight_bytes(GPT2_SMALL, ITEMSIZE[dtype]):,}",
            f"{per_token:,} ({per_token / 1024:.0f} KiB)",
            f"{per_token * 1024:,} ({mib(per_token * 1024)} MiB)",
        ])
    return (
        "From `benchmarks/costs.py` for GPT-2 small (B=1); formulas, not measurements.\n\n"
        + table(["dtype", "weight bytes read per step", "KV bytes per token",
                 "KV bytes at 1024 tokens"], rows)
    )


def _decode_table(payload: dict[str, Any], variants: tuple[str, ...]) -> str:
    by_t: dict[int, dict[str, dict[str, Any]]] = {}
    for record in suite_records(payload, "decode"):
        by_t.setdefault(record["context_len"], {})[record["variant"]] = record
    header = ["T"] + [f"{v} ms (roofline)" for v in variants]
    if "graph" in variants:
        header.append("graph floor + bandwidth ms")
    header.append("sequences match")
    rows = []
    for t, row in sorted(by_t.items()):
        cells: list[Any] = [t]
        cells += [f"{ms(row[v]['median_s'])} ({ms(row[v]['roofline_per_token_s'])})" for v in variants]
        if "graph" in variants:
            cells.append(ms(row["graph"]["graph_floor_plus_bandwidth_per_token_s"]))
        match = row[variants[0]]["sequences_match"]
        if "graph" in row:
            per = ", ".join(f"{v} {row[v]['matches_uncached']}" for v in variants[1:])
            match = f"{match} ({per})"
        cells.append(match)
        rows.append(cells)
    return table(header, rows)


def section_decode_gpu(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = gpu_file(dtype)
        payload = load(results, name)
        steps = suite_records(payload, "decode")[0]["steps"]
        parts.append(
            f"### {dtype}\n\n{source(name, payload)} B=1, {steps} decode steps after an untimed "
            "prefill; time per token is the median.\n\n"
            + _decode_table(payload, ("uncached", "dynamic", "static", "graph"))
        )
    return "\n\n".join(parts)


def section_decode_cpu(results: Path) -> str:
    name = "decode_cpu_float32.json"
    payload = load(results, name)
    steps = suite_records(payload, "decode")[0]["steps"]
    return (
        f"{source(name, payload)} float32, B=1, {steps} decode steps.\n\n"
        + _decode_table(payload, ("uncached", "dynamic", "static"))
    )


def section_breakdown(results: Path) -> str:
    rows = []
    for dtype in DTYPES:
        data = time_breakdown(results, dtype)
        eager, graph = data["eager"], data["graph"]
        rows.append([dtype, "eager static", ms(eager["measured"]),
                     f"CPU overhead {ms(eager['cpu_overhead'])}, rest {ms(eager['rest'])}"])
        rows.append([dtype, "graph", ms(graph["measured"]),
                     f"kernel floor {ms(graph['kernel_floor'])}, weights {ms(graph['weights'])}, "
                     f"cache ({graph['cache_capacity']} slots) {ms(graph['cache'])}, "
                     f"residual {ms(graph['residual'])}"])
    bandwidth = calibration_values(results)["cuda_read_GBs"]
    return (
        "B=1, T=512, ms per token. Sources: `all_cuda_<dtype>.json` (decode and overhead suites) "
        f"and `calibration.json` (read bandwidth {bandwidth:.1f} GB/s). Eager: CPU overhead is the "
        "structural model in eager static mode. Graph: kernel floor is the structural model in the "
        "graph; weights and cache are bytes over the read bandwidth; residual is what remains.\n\n"
        + table(["dtype", "path", "measured", "parts"], rows)
    )


def section_overhead(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = gpu_file(dtype)
        payload = load(results, name)
        rows = [
            [r["model"], r["variant"], ms(r["total_median_s_per_token"]),
             f"{ms(r['total_p25_s_per_token'])}–{ms(r['total_p75_s_per_token'])}",
             ms(r["host_median_s_per_token"]), f"{r['host_to_total_ratio']:.2f}", r["ops"]["total"]]
            for r in suite_records(payload, "overhead")
        ]
        context = suite_records(payload, "overhead")[0]["context_len"]
        parts.append(f"### {dtype}\n\n{source(name, payload)} B=1, T={context}.\n\n" + table(
            ["model", "variant", "ms/token", "p25–p75", "host ms/token", "host/total", "aten ops"],
            rows,
        ))
    return "\n\n".join(parts)


def section_batch(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = gpu_file(dtype)
        payload = load(results, name)
        rows = []
        for r in suite_records(payload, "batch"):
            if r.get("out_of_memory"):
                rows.append([r["batch_size"], "out of memory", "", "", "", "", "", "", ""])
                continue
            rows.append([
                r["batch_size"], r["variant"], f"{r['tokens_per_s']:.0f}", ms(r["median_s"] / r["steps"]),
                ms(r["roofline_per_step_s"]), mib(r["max_memory_allocated"]),
                mib(r["max_memory_reserved"]), r["cache_capacity"], r["sequences_match"],
            ])
        parts.append(f"### {dtype}\n\n{source(name, payload)} T=512, 16 decode steps.\n\n" + table(
            ["B", "variant", "tokens/s", "ms/step", "roofline ms/step", "max allocated MiB",
             "max reserved MiB", "cache capacity", "sequences match"],
            rows,
        ))
    return "\n\n".join(parts)


def section_capacity(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = gpu_file(dtype)
        payload = load(results, name)
        rows = [
            [r["cache_capacity"], ms(r["median_s"]), f"{ms(r['p25_s'])}–{ms(r['p75_s'])}",
             ms(r["roofline_per_token_s"]), mib(r["step_bytes"]), mib(r["cache_nbytes"]),
             mib(r["cache_used_nbytes"]), r["sequences_match"]]
            for r in sorted(suite_records(payload, "capacity"), key=lambda r: r["cache_capacity"])
        ]
        parts.append(f"### {dtype}\n\n{source(name, payload)} Graph only, B=8, T=64, 16 steps.\n\n"
                     + table(["capacity", "ms/token", "p25–p75", "roofline ms", "step MiB",
                              "cache MiB", "used MiB", "sequences match"], rows))
    return "\n\n".join(parts)


def _prefill_table(payload: dict[str, Any]) -> str:
    rows = [
        [r["prompt_len"], r["variant"], ms(r["median_s"]), f"{r['achieved_TFLOPs']:.3f}",
         f"{r['peak_fraction']:.2f}", ms(r["roofline_s"])]
        for r in suite_records(payload, "prefill")
    ]
    return table(["T", "variant", "ms", "TFLOP/s", "peak fraction", "roofline ms"], rows)


def section_prefill(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = gpu_file(dtype)
        payload = load(results, name)
        parts.append(f"### GPU, {dtype}\n\n{source(name, payload)} B=1.\n\n" + _prefill_table(payload))
    name = "prefill_cpu_float32.json"
    payload = load(results, name)
    parts.append(f"### CPU, float32\n\n{source(name, payload)} B=1.\n\n" + _prefill_table(payload))
    return "\n\n".join(parts)


def section_memory(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = gpu_file(dtype)
        payload = load(results, name)
        rows = []
        for r in suite_records(payload, "memory"):
            if r.get("out_of_memory"):
                rows.append([r["batch_size"], r["variant"], "out of memory", "", "", ""])
                continue
            last = r["snapshots"][-1]
            rows.append([r["batch_size"], r["variant"], mib(r["max_memory_allocated"]),
                         mib(r["max_memory_reserved"]), mib(last["cache_nbytes"]),
                         mib(last["cache_used_nbytes"])])
        record = suite_records(payload, "memory")[0]
        parts.append(
            f"### {dtype}\n\n{source(name, payload)} Prompt {record['prompt_len']} tokens plus "
            f"{record['new_tokens']} generated, untimed.\n\n"
            + table(["B", "variant", "max allocated MiB", "max reserved MiB", "final cache MiB",
                     "final used MiB"], rows)
        )
    return "\n\n".join(parts)


def section_roofline(results: Path) -> str:
    rows = [
        [p["dtype"], p["kind"], p["label"], f"{p['intensity']:.2f}", f"{p['achieved_TFLOPs']:.3f}"]
        for p in roofline_points(results)
    ]
    return (
        "Derived from the batch suite (decode, T=512) and the prefill suite (uncached, B=1) of "
        "`all_cuda_<dtype>.json`: cost-model FLOPs and bytes, measured time.\n\n"
        + table(["dtype", "kind", "point", "FLOP/byte", "achieved TFLOP/s"], rows)
    )


def section_isolation(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = f"graph_isolation_cuda_{dtype}.json"
        payload = load(results, name)
        suite = payload["suites"]["graph_isolation"]
        modes = suite["modes"]
        by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for r in suite["records"]:
            by_key.setdefault((r["model"], r["context_len"]), {})[r["mode"]] = r
        rows = [
            [model, t] + [ms(row[m]["median_s"]) for m in modes]
            + [all(row[m]["repeats_identical"] for m in modes)]
            for (model, t), row in sorted(by_key.items())
        ]
        blocks = [
            [b["mode"], b["before"]["clocks.sm"] if b["before"] else "n/a",
             b["after"]["clocks.sm"] if b["after"] else "n/a",
             b.get("started_at", "not recorded"), b.get("finished_at", "not recorded")]
            for b in suite["blocks"]
        ]
        parts.append(
            f"### {dtype}\n\n{source(name, payload)} B=1, 16 replayed steps, capacity 1024; "
            "ms per token (median).\n\n"
            + table(["model", "T"] + modes + ["repeats identical"], rows)
            + "\n\n"
            + table(["block", "SM MHz before", "SM MHz after", "started", "finished"], blocks)
        )
    return "\n\n".join(parts)


def section_shapes(results: Path) -> str:
    parts = []
    for dtype in DTYPES:
        name = f"shapes_cuda_{dtype}.json"
        payload = load(results, name)
        suite = payload["suites"]["shapes"]
        rows = [
            [r["family"], r["layer"], f"{r['in_features']}→{r['out_features']}", r["batch_size"],
             f"{r['median_s'] * 1e6:.2f}", f"{r['effective_bandwidth_GBs']:.1f}",
             f"{r['effective_TFLOPs']:.3f}"]
            for r in suite["records"]
        ]
        curve = [
            [r["size_mib"], f"{r['median_s'] * 1e6:.2f}", f"{r['effective_bandwidth_GBs']:.1f}"]
            for r in suite["bandwidth_records"]
        ]
        parts.append(
            f"### {dtype}\n\n{source(name, payload)} 20 calls per CUDA graph; one weight reused, so "
            "weights that fit in L2 can exceed DRAM bandwidth.\n\n"
            + table(["family", "layer", "in→out", "B", "µs/call", "GB/s", "TFLOP/s"], rows)
            + "\n\nRead bandwidth, `x.sum()` on float32:\n\n"
            + table(["MiB", "µs/call", "GB/s"], curve)
        )
    return "\n\n".join(parts)


def collect_test_counts() -> dict[str, int]:
    """Count tests with ``pytest --collect-only`` for the default and integration runs."""
    counts = {}
    for key, extra in (("fast", []), ("integration", ["-m", "integration"])):
        output = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", *extra],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout
        match = re.search(r"(\d+)(?:/\d+)? tests? collected", output)
        if match is None:
            raise RuntimeError(f"could not read the pytest collection count:\n{output}")
        counts[key] = int(match.group(1))
    return counts


def build_summary(results: Path = RESULTS_DIR, test_counts: dict[str, int] | None = None) -> str:
    builders = {
        "Provenance": lambda: section_provenance(results),
        "Correctness": lambda: section_correctness(test_counts),
        "Calibration": lambda: section_calibration(results),
        "Cost model constants": section_cost_model,
        "Decode on the GPU": lambda: section_decode_gpu(results),
        "Decode on the CPU": lambda: section_decode_cpu(results),
        "Time breakdown": lambda: section_breakdown(results),
        "Overhead": lambda: section_overhead(results),
        "Batch": lambda: section_batch(results),
        "Capacity": lambda: section_capacity(results),
        "Prefill": lambda: section_prefill(results),
        "Memory": lambda: section_memory(results),
        "Roofline points": lambda: section_roofline(results),
        "Graph isolation": lambda: section_isolation(results),
        "Shapes": lambda: section_shapes(results),
    }
    body = [
        "# Benchmark results",
        "",
        "Generated by `uv run python -m benchmarks.summary` from the JSON files in "
        "`benchmarks/results/`. Do not edit by hand. Times are medians; p25–p75 bands are in the "
        "JSON files. Methodology: ADR-0009. Variability between runs: ADR-0013.",
    ]
    for name in SECTIONS:
        body += ["", f"## {name}", "", builders[name]()]
    return "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_MD)
    parser.add_argument("--no-test-counts", action="store_true")
    args = parser.parse_args()
    counts = None if args.no_test_counts else collect_test_counts()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(build_summary(args.results, counts).encode("utf-8"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
