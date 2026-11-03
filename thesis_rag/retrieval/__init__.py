"""Retrieval variants and composition helpers."""

from thesis_rag.retrieval.candidates import (
    CandidateBatch,
    CandidateDocument,
    CandidateMatch,
    CandidateRanking,
    ComponentScore,
    RankedCandidate,
    RankTransition,
)
from thesis_rag.retrieval.config import ModernPipelineConfig, ModernSystemSettings
from thesis_rag.retrieval.pipeline import PipelineComponents
from thesis_rag.retrieval.system import ModernRetrievalSystem

__all__ = [
    "CandidateBatch",
    "CandidateDocument",
    "CandidateMatch",
    "CandidateRanking",
    "ComponentScore",
    "ModernPipelineConfig",
    "ModernRetrievalSystem",
    "ModernSystemSettings",
    "PipelineComponents",
    "RankTransition",
    "RankedCandidate",
]
