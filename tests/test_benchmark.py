"""Tests for the benchmark utility."""
import pytest
from chunkops import benchmark
from chunkops.benchmark import BenchmarkResult, StrategyStats


def test_benchmark_returns_result(medium_doc):
    result = benchmark([medium_doc], strategies=["fixed", "recursive"])
    assert isinstance(result, BenchmarkResult)


def test_benchmark_best_strategy_is_valid(medium_doc):
    result = benchmark([medium_doc], strategies=["fixed", "recursive", "structural"])
    assert result.best_strategy in ["fixed", "recursive", "structural"]


def test_benchmark_stats_count_matches_strategies(medium_doc):
    result = benchmark([medium_doc], strategies=["fixed", "recursive"])
    assert len(result.stats) == 2


def test_benchmark_coherence_metric(medium_doc):
    result = benchmark([medium_doc], strategies=["recursive", "structural"], metric="coherence")
    assert result.metric == "coherence"
    for s in result.stats:
        assert 0.0 <= s.coherence_score <= 1.0


def test_benchmark_chunk_count_metric(medium_doc):
    result = benchmark([medium_doc], strategies=["fixed", "recursive"], metric="chunk_count")
    assert result.metric == "chunk_count"


def test_benchmark_multiple_docs(medium_doc, markdown_doc):
    result = benchmark([medium_doc, markdown_doc], strategies=["recursive"])
    assert result.doc_count == 2
    assert result.stats[0].chunk_count > 0


def test_benchmark_stats_fields(medium_doc):
    result = benchmark([medium_doc], strategies=["recursive"])
    s = result.stats[0]
    assert s.strategy == "recursive"
    assert s.chunk_count > 0
    assert s.avg_tokens > 0
    assert s.min_tokens > 0
    assert s.max_tokens >= s.min_tokens
    assert s.elapsed_ms >= 0


def test_benchmark_report_runs_without_error(medium_doc, capsys):
    result = benchmark([medium_doc], strategies=["fixed", "recursive"])
    result.report()
    captured = capsys.readouterr()
    assert "strategy" in captured.out
    assert "recursive" in captured.out


def test_benchmark_best_method(medium_doc):
    result = benchmark([medium_doc], strategies=["fixed", "recursive"])
    best = result.best()
    assert isinstance(best, StrategyStats)
    assert best.strategy == result.best_strategy


def test_benchmark_fixed_has_more_boundary_breaks_than_recursive(medium_doc):
    result = benchmark([medium_doc], strategies=["fixed", "recursive"])
    fixed = next(s for s in result.stats if s.strategy == "fixed")
    recursive = next(s for s in result.stats if s.strategy == "recursive")
    assert fixed.boundary_breaks >= recursive.boundary_breaks


def test_benchmark_empty_strategies_raises(medium_doc):
    with pytest.raises((RuntimeError, Exception)):
        benchmark([medium_doc], strategies=[])
