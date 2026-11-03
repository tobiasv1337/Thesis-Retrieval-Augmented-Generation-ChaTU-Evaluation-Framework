import pytest

from thesis_rag.eval.known_targets import distinct_page_ranking, known_target_scores


def test_aliases_collapse_but_unmapped_predictions_keep_failure_positions():
    ranking = distinct_page_ranking([None, "page:a", "page:a", None, "page:b"], top_k=4)
    assert ranking == [None, "page:a", None, "page:b"]
    scores = known_target_scores(ranking, {"page:b"}, cutoffs=[1, 3, 4, 20])
    assert scores["3"] == {"hit": 0.0, "mrr": 0.0}
    assert scores["4"] == {"hit": 1.0, "mrr": 0.25}
    assert scores["20"] == scores["4"]


def test_counterparts_change_target_acceptance_without_compressing_ranks():
    ranking = ["unrelated_de", "unrelated_en", "answer_en"]
    families = {
        "unrelated_de": "family:u",
        "unrelated_en": "family:u",
        "answer_de": "family:a",
        "answer_en": "family:a",
    }
    exact = known_target_scores(ranking, {"answer_de"}, cutoffs=[1, 2, 3])
    counterpart = known_target_scores(
        ranking, {"answer_de"}, cutoffs=[1, 2, 3], counterpart_families=families
    )
    assert exact["3"]["hit"] == 0.0
    assert counterpart["2"]["hit"] == 0.0
    assert counterpart["3"] == {"hit": 1.0, "mrr": 1 / 3}


def test_multiple_historical_targets_and_no_results():
    assert known_target_scores(["b"], {"a", "b"}, cutoffs=[1])["1"]["hit"] == 1.0
    assert known_target_scores([], {"a"}, cutoffs=[1, 20])["20"] == {
        "hit": 0.0,
        "mrr": 0.0,
    }
    with pytest.raises(ValueError, match="reference Page"):
        known_target_scores([], set(), cutoffs=[1])
