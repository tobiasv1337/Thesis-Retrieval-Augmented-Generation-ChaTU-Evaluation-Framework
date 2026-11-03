from types import SimpleNamespace

import pandas as pd

from thesis_rag.eval import retrieval_analysis as analysis
from thesis_rag.eval.suite_report import SuiteReportConfig


def test_metric_highlights_use_raw_values_directions_ties_and_groups():
    values = pd.DataFrame({
        "scope": ["full", "full", "small", "small"],
        "quality": [0.758599, 0.758514, 0.9, 0.9],
        "latency": [170.0, 160.0, None, 200.0],
        "missing": [float("nan")] * 4,
    })
    assert analysis._best_metric_cells(values, minimize=("latency",), groups=("scope",)) == {
        (0, "quality"), (2, "quality"), (3, "quality"),
        (1, "latency"), (3, "latency"),
    }


def test_latex_highlights_preserve_escaping_and_plain_data():
    frame = pd.DataFrame({"System_name": ["A&B", "C_2"], "nDCG": [0.8, 0.9]})
    before = frame.to_csv(index=False)
    frame.attrs["bold_cells"] = {(0, "System_name"), (1, "nDCG")}
    latex = analysis._latex_table(frame).decode()
    assert r"\bfseries A\&B" in latex
    assert r"\bfseries 0.9000" in latex
    assert r"\bfseries 0.8000" not in latex
    assert r"System\_name" in latex and r"C\_2" in latex
    assert frame.to_csv(index=False) == before


def test_comparison_marks_selection_separately_from_metric_winners(monkeypatch):
    rows = [dict(
        experiment_id=key, judgment_view="judge:hybrid", split="development", stream="items",
        target_level=target, metric=metric, cutoff=20, estimate=value,
        ci_lower=value - 0.01, ci_upper=value + 0.01,
    ) for key, chunk, page in [("chosen", 0.758599, 0.80), ("other", 0.758514, 0.81)]
        for target, metric, value in [("chunk", "ndcg", chunk), ("page", "ndcg", page),
                                      ("page", "counterpart_family_ndcg", page)]]
    tables = {
        "performance_by_cutoff": pd.DataFrame(rows),
        "efficiency_summary": pd.DataFrame([
            dict(experiment_id="chosen", index_byte_size=1024, retrieval_p95_ms=172),
            dict(experiment_id="other", index_byte_size=1024, retrieval_p95_ms=166),
        ]),
        "experiment_details": pd.DataFrame(),
        "selection_summary": pd.DataFrame([
            dict(comparison_id="systems", selected_experiments="chosen"),
            dict(comparison_id="unrelated", selected_experiments="other"),
        ]),
    }
    report_config = SuiteReportConfig(report_config_id="fixture")
    summaries = analysis._comparison_summary_tables(
        tables, comparisons=[SimpleNamespace(
            comparison_id="systems", judgment_view_label="judge:hybrid", stream="items",
            split="development", member_experiment_ids=["chosen", "other"], target_space_id="chunk",
        )], primary_view="judge:hybrid", primary_cutoff=20, report_config=report_config, pd=pd,
    )
    frame = summaries["comparison_summary__systems"]
    bold = frame.attrs["bold_cells"]
    assert (0, "Experiment") in bold and (1, "Experiment") not in bold
    assert (0, "Chunk nDCG@20") in bold and (1, "Chunk nDCG@20") not in bold
    assert (1, "Page nDCG@20") in bold and (0, "Page nDCG@20") not in bold
    assert (1, "p95 latency (ms)") in bold
    monkeypatch.setattr(analysis, "TABLE_FILES", ())
    monkeypatch.setattr(analysis, "_publication_tables", lambda *args, **kwargs: {})
    payloads = analysis._table_payloads(tables, comparison_tables=summaries, report_config=report_config, pd=pd)
    latex = payloads["table_comparison_summary__systems_tex"][1].decode()
    csv = payloads["table_comparison_summary__systems_csv"][1].decode()
    assert "Page-family nDCG@20" in latex and r"\bfseries 0.8100" in latex
    assert "Index size" not in latex and "Index size" in csv
    assert r"\bfseries" not in csv


def test_rrf_label_distinguishes_constant_from_depth():
    assert analysis._display_name("content_rrf_dedup_k30") == "RRF k = 30"
