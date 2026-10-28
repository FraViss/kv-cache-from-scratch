from pathlib import Path

from benchmarks.plots import FIGURE_NAMES, make_figures
from benchmarks.results_io import RESULTS_DIR, time_breakdown
from benchmarks.summary import SECTIONS, build_summary


def test_plots_write_every_figure_from_committed_results(tmp_path: Path) -> None:
    written = make_figures(RESULTS_DIR, tmp_path)
    assert [path.name for path in written] == list(FIGURE_NAMES)
    for path in written:
        assert path.parent == tmp_path
        assert path.read_bytes().startswith(b"\x89PNG")


def test_summary_has_every_section_and_names_its_sources() -> None:
    markdown = build_summary(RESULTS_DIR, test_counts={"fast": 1, "integration": 2})
    for name in SECTIONS:
        assert f"\n## {name}\n" in markdown
    for json_file in RESULTS_DIR.glob("*.json"):
        assert json_file.name in markdown
    assert "1 fast tests" in markdown
    assert "Source: `benchmarks/results/" in markdown


def test_summary_without_test_counts_skips_that_line() -> None:
    markdown = build_summary(RESULTS_DIR, test_counts=None)
    assert "Test counts collected by pytest" not in markdown


def test_time_breakdown_parts_add_up_to_the_measured_step() -> None:
    for dtype in ("float32", "bfloat16"):
        data = time_breakdown(RESULTS_DIR, dtype)
        eager, graph = data["eager"], data["graph"]
        assert abs(eager["cpu_overhead"] + eager["rest"] - eager["measured"]) < 1e-12
        parts = graph["kernel_floor"] + graph["weights"] + graph["cache"] + graph["residual"]
        assert abs(parts - graph["measured"]) < 1e-12
        assert graph["cache_capacity"] == 1024
