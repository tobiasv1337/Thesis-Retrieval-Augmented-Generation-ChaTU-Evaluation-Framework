"""Normal Page-QREL evaluation preserves immutable native Page ranking slots."""

from types import SimpleNamespace as NS

import pytest

from tests.thesis_rag.test_known_target_report import completed_manual_suite
from tests.thesis_rag.test_manual_benchmark import manual_workspace
from thesis_rag.eval.relevance import (
    _configured_raw_rankings, _load_retrieval_results, _ranked_dcg,
)
from thesis_rag.eval.relevance_config import load_relevance_config
from thesis_rag.eval.suite import ExperimentSuitePlanJob, load_pinned_retrievals
from thesis_rag.schema import hash_file


def test_normal_qrels_preserve_unmapped_nlu_slots(completed_manual_suite):
    root = completed_manual_suite.parent
    suite = ExperimentSuitePlanJob(completed_manual_suite, repo_root=root).run().manifest
    config = load_relevance_config("configs/thesis/relevance.yaml")
    for experiment, _, _, path in load_pinned_retrievals(suite, root):
        before = hash_file(path)
        results = _load_retrieval_results([path])
        policy = experiment.final_evaluation.model_copy(update={"include_in_pool": True})
        selection = NS(experiment=experiment.model_copy(update={"final_evaluation": policy}))
        for result in results:
            native_rank = 2 if experiment.experiment_id.startswith("backup") else 1
            assert result.items[0].rank == 1
            for purpose in ("pool", "metrics"):
                ranking, = _configured_raw_rankings(result, config, selection=selection, purpose=purpose)
                assert ranking.items[0].rank == native_rank
                assert ranking.evaluable_up_to_k == 20
                assert ranking.ranking_complete
                assert _ranked_dcg(ranking.items, [3]) == pytest.approx(
                    7 if native_rank == 1 else 7 / 1.584962500721156
                )
            assert result.items[0].rank == 1
            bad = result.model_copy(update={"debug": {"page_ranking": {}}})
            with pytest.raises(ValueError):
                _configured_raw_rankings(bad, config, selection=selection)
        assert hash_file(path) == before
