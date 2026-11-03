"""Shared agreement metrics for ordinal relevance-score mappings."""

from __future__ import annotations

from collections.abc import Hashable, Mapping

from thesis_rag.artifacts import JsonDict


def _precision_recall_f1(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> JsonDict:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else None
    )
    recall = true_positive / recall_denominator if recall_denominator else None
    if true_positive:
        if precision is None or recall is None:
            raise AssertionError("positive class has no metric denominator")
        f1: float | None = 2 * precision * recall / (precision + recall)
    elif false_positive or false_negative:
        f1 = 0.0
    else:
        f1 = None
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_agreement(
    reference: Mapping[Hashable, int],
    candidate: Mapping[Hashable, int],
    *,
    reference_label: str,
    candidate_label: str,
) -> JsonDict:
    """Return ordinal and binary agreement metrics on common score keys."""

    keys = sorted(set(reference) & set(candidate), key=repr)
    confusion = {
        str(reference_score): {
            str(candidate_score): 0 for candidate_score in range(4)
        }
        for reference_score in range(4)
    }
    reference_marginals = [0] * 4
    candidate_marginals = [0] * 4
    exact = 0
    within_one = 0
    binary = 0
    absolute_delta = 0
    weighted_observed = 0.0
    for key in keys:
        reference_score = reference[key]
        candidate_score = candidate[key]
        if reference_score not in range(4) or candidate_score not in range(4):
            raise ValueError("relevance agreement scores must be between 0 and 3")
        confusion[str(reference_score)][str(candidate_score)] += 1
        reference_marginals[reference_score] += 1
        candidate_marginals[candidate_score] += 1
        delta = abs(reference_score - candidate_score)
        exact += int(delta == 0)
        within_one += int(delta <= 1)
        binary += int((reference_score >= 2) == (candidate_score >= 2))
        absolute_delta += delta
        weighted_observed += (delta / 3) ** 2

    count = len(keys)
    cohen_kappa: float | None = None
    weighted_kappa: float | None = None
    if count:
        observed_exact = exact / count
        expected_exact = sum(
            reference_marginals[score]
            / count
            * candidate_marginals[score]
            / count
            for score in range(4)
        )
        if expected_exact < 1.0:
            cohen_kappa = (observed_exact - expected_exact) / (
                1.0 - expected_exact
            )
        observed = weighted_observed / count
        expected = sum(
            (abs(reference_score - candidate_score) / 3) ** 2
            * reference_marginals[reference_score]
            / count
            * candidate_marginals[candidate_score]
            / count
            for reference_score in range(4)
            for candidate_score in range(4)
        )
        if expected > 0:
            weighted_kappa = 1.0 - observed / expected

    per_grade: dict[str, JsonDict] = {}
    observed_grade_f1: list[float] = []
    for grade in range(4):
        metrics = _precision_recall_f1(
            true_positive=confusion[str(grade)][str(grade)],
            false_positive=sum(
                confusion[str(reference_score)][str(grade)]
                for reference_score in range(4)
                if reference_score != grade
            ),
            false_negative=sum(
                confusion[str(grade)][str(candidate_score)]
                for candidate_score in range(4)
                if candidate_score != grade
            ),
        )
        metrics["reference_support"] = reference_marginals[grade]
        metrics["candidate_predictions"] = candidate_marginals[grade]
        per_grade[str(grade)] = metrics
        if reference_marginals[grade] or candidate_marginals[grade]:
            observed_grade_f1.append(float(metrics["f1"] or 0.0))

    binary_metrics = _precision_recall_f1(
        true_positive=sum(
            confusion[str(reference_score)][str(candidate_score)]
            for reference_score in (2, 3)
            for candidate_score in (2, 3)
        ),
        false_positive=sum(
            confusion[str(reference_score)][str(candidate_score)]
            for reference_score in (0, 1)
            for candidate_score in (2, 3)
        ),
        false_negative=sum(
            confusion[str(reference_score)][str(candidate_score)]
            for reference_score in (2, 3)
            for candidate_score in (0, 1)
        ),
    )
    reference_positive = sum(reference_marginals[2:])
    candidate_positive = sum(candidate_marginals[2:])
    binary_metrics["reference_positive_support"] = reference_positive
    binary_metrics["candidate_positive_predictions"] = candidate_positive
    if count:
        binary_expected = (
            reference_positive / count * candidate_positive / count
            + (count - reference_positive)
            / count
            * (count - candidate_positive)
            / count
        )
        binary_metrics["cohen_kappa"] = (
            (binary / count - binary_expected) / (1.0 - binary_expected)
            if binary_expected < 1.0
            else None
        )
    else:
        binary_metrics["cohen_kappa"] = None

    return {
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "reference_judgments": len(reference),
        "candidate_judgments": len(candidate),
        "paired_judgments": count,
        "exact_agreement": exact / count if count else None,
        "within_one_grade_agreement": within_one / count if count else None,
        "binary_relevance_agreement": binary / count if count else None,
        "mean_absolute_grade_delta": absolute_delta / count if count else None,
        "cohen_kappa": cohen_kappa,
        "quadratic_weighted_kappa": weighted_kappa,
        "macro_f1": (
            sum(observed_grade_f1) / len(observed_grade_f1)
            if observed_grade_f1
            else None
        ),
        "macro_f1_grades": [
            grade
            for grade in range(4)
            if reference_marginals[grade] or candidate_marginals[grade]
        ],
        "binary_relevance": binary_metrics,
        "grade_3": per_grade["3"],
        "per_grade": per_grade,
        "confusion_matrix_reference_rows_candidate_columns": confusion,
    }
