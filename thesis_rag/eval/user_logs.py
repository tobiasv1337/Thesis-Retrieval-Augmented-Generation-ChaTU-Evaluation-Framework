"""Private CSV logs as versioned, source-anchor-free evaluation questions."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date, datetime, time
import math
from pathlib import Path
import re
import unicodedata
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo

from pydantic import Field, TypeAdapter, model_validator
import yaml

from thesis_rag.artifacts import (
    ArtifactModel,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.eval.artifact_io import (
    read_json_object,
    write_json_object_atomic,
    write_jsonl_atomic,
)
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.schema import (
    QUESTION_SCHEMA_VERSION,
    USER_LOG_IMPORT_CONFIG_SCHEMA_VERSION,
    USER_LOG_TURN_IMPORT_CONFIG_SCHEMA_VERSION,
    USER_LOG_QUESTION_SET_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
)


CSV_COLUMNS = (
    "user_id", "timestamp", "bot", "turn_number", "language", "text",
    "button_click", "intent", "bot_intent",
)
TURN_CSV_COLUMNS = (
    "session_id", "timestamp", "date", "language", "text", "intent",
    "confidence", "bot_intent", "bot_confidence", "reply_bot", "user_turn_index",
)
ARTIFACT_FILES = {
    "questions": "questions.jsonl", "review": "review.json",
    "report": "report.json",
}
SPLITS = ("development", "validation", "test")


class UserLogImportConfig(ArtifactModel):
    schema_version: Literal[USER_LOG_IMPORT_CONFIG_SCHEMA_VERSION] = (
        USER_LOG_IMPORT_CONFIG_SCHEMA_VERSION
    )
    dataset_id: NonEmptyStr = "sekretariat_user_logs"
    delimiter: Literal[",", ";", "\t"] = ","
    intent: NonEmptyStr = "sekretariat"
    language_labels: dict[NonEmptyStr, Literal["de", "en"]] = Field(
        default_factory=lambda: {"deutsch": "de", "english": "en", "de": "de", "en": "en"},
    )
    excluded_exact_texts: list[NonEmptyStr] = Field(default_factory=list)
    max_characters: PositiveInt | None = None
    long_review_characters: PositiveInt = 1000
    seed: NonNegativeInt = 14731
    split_percentages: dict[str, NonNegativeInt] = Field(
        default_factory=lambda: {"development": 40, "validation": 20, "test": 40},
    )

    @model_validator(mode="after")
    def valid_split(self) -> Self:
        if (
            set(self.split_percentages) != set(SPLITS)
            or sum(self.split_percentages.values()) != 100
        ):
            raise ValueError(
                "split_percentages must define development/validation/test and sum to 100",
            )
        if not self.language_labels:
            raise ValueError("language_labels must not be empty")
        return self


class UserLogTurnImportConfig(UserLogImportConfig):
    """Explicit contract for the session export; v1 identities stay unchanged."""

    schema_version: Literal[USER_LOG_TURN_IMPORT_CONFIG_SCHEMA_VERSION] = (
        USER_LOG_TURN_IMPORT_CONFIG_SCHEMA_VERSION
    )
    csv_schema: Literal["first_user_turns_v1"] = "first_user_turns_v1"
    user_turn_index: PositiveInt = 2
    reply_bot: NonEmptyStr = "sekretariat"
    start_date_inclusive: NonEmptyStr
    end_date_exclusive: NonEmptyStr
    timezone: NonEmptyStr = "Europe/Berlin"
    strip_leading_bot_prefix: Literal["@sekretariat"] = "@sekretariat"

    @model_validator(mode="after")
    def valid_window(self) -> Self:
        for value in (self.start_date_inclusive, self.end_date_exclusive):
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("date bounds must use YYYY-MM-DD")
        if self.start_date_inclusive >= self.end_date_exclusive:
            raise ValueError("start_date_inclusive must precede end_date_exclusive")
        try:
            ZoneInfo(self.timezone)
        except (KeyError, ValueError) as exc:
            raise ValueError("unknown import timezone") from exc
        return self

    def timestamp_bounds(self) -> tuple[float, float]:
        zone = ZoneInfo(self.timezone)
        return tuple(
            datetime.combine(date.fromisoformat(value), time.min, tzinfo=zone).timestamp()
            for value in (self.start_date_inclusive, self.end_date_exclusive)
        )


UserLogConfig = Annotated[
    UserLogImportConfig | UserLogTurnImportConfig, Field(discriminator="schema_version"),
]


def _source_exclusion(row: dict[str, str], config: UserLogConfig) -> str:
    if isinstance(config, UserLogTurnImportConfig):
        if row["intent"].strip() != config.intent or row["reply_bot"].strip() != config.reply_bot:
            return "other_bot"
        if not re.fullmatch(r"\d+", row["user_turn_index"].strip()):
            return "invalid_turn"
        if int(row["user_turn_index"]) != config.user_turn_index:
            return "not_selected_turn"
        try:
            timestamp = float(row["timestamp"])
        except ValueError:
            return "invalid_timestamp"
        if not math.isfinite(timestamp):
            return "invalid_timestamp"
        lower, upper = config.timestamp_bounds()
        if not lower <= timestamp < upper:
            return "outside_time_window"
    else:
        if row["bot"].strip() != "user":
            return "not_user"
        if row["intent"].strip() != config.intent:
            return "other_bot"
        if not re.fullmatch(r"\d+", row["turn_number"].strip()):
            return "invalid_turn"
        if int(row["turn_number"]) != 1:
            return "not_turn_one"
        if row["button_click"].strip().casefold() not in {"true", "false"}:
            return "invalid_button_click"
        if row["button_click"].strip().casefold() == "true":
            return "button_click"
    return ""


def normalize_log_text(text: str) -> str:
    """Conservative exact deduplication; never rewrite the actual query."""
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def _noise_key(text: str) -> str:
    return normalize_log_text(text).strip(" .!?…")


def _split(text: str, config: UserLogImportConfig) -> str:
    # Language is deliberately absent: the same text with different UI signals
    # must remain in one split. Hash assignment is stable when filters change.
    digest = hash_jsonable([config.seed, normalize_log_text(text)]).split(":")[1]
    bucket = int(digest, 16) % 100
    boundary = 0
    for name in SPLITS:
        boundary += config.split_percentages[name]
        if bucket < boundary:
            return name
    raise AssertionError("validated split percentages do not cover bucket")


def _query_id(text: str, language: str, config: UserLogImportConfig) -> str:
    return stable_id("user_log_query", config.dataset_id, normalize_log_text(text), language)


def _flags(text: str, config: UserLogImportConfig) -> list[str]:
    flags = []
    if len(text.strip()) <= 30:
        flags.append("short_query")
    if len(text) > config.long_review_characters:
        flags.append("long_query")
    if re.search(r"[^\s@]+@[^\s@]+\.[^\s@]+", text):
        flags.append("possible_email")
    if re.search(r"(?<!\w)\+?\d[\d ()/.-]{6,}\d(?!\w)", text):
        flags.append("possible_identifier_or_phone")
    return flags


def _set_id(
    source_hash: str, config: UserLogImportConfig,
    exclusions: dict[str, str], status: str,
) -> str:
    return stable_id(
        "user_log_set", USER_LOG_QUESTION_SET_SCHEMA_VERSION,
        source_hash, config.to_dict(), exclusions, status,
    )


class UserLogQuestionSetManifest(ArtifactModel):
    schema_version: Literal[USER_LOG_QUESTION_SET_SCHEMA_VERSION] = (
        USER_LOG_QUESTION_SET_SCHEMA_VERSION
    )
    question_set_id: NonEmptyStr
    status: Literal["imported"] = "imported"
    data_classification: Literal["private"] = "private"
    source_hash: Sha256
    config: UserLogConfig
    exclusions: dict[NonEmptyStr, NonEmptyStr]
    question_count: NonNegativeInt
    artifact_hashes: dict[NonEmptyStr, Sha256]

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        if self.question_set_id != _set_id(self.source_hash, self.config, self.exclusions, self.status):
            raise ValueError("log question-set identity differs from its inputs")
        if set(self.artifact_hashes) != set(ARTIFACT_FILES):
            raise ValueError("log question-set artifacts are incomplete")
        return self


def verify_user_log_question_set(
    path: Path,
) -> tuple[UserLogQuestionSetManifest, tuple[QuestionRecord, ...]]:
    manifest = UserLogQuestionSetManifest.from_dict(read_json_object(path, error_type=ValueError))
    for name, filename in ARTIFACT_FILES.items():
        artifact = path.parent / filename
        if not artifact.is_file() or hash_file(artifact) != manifest.artifact_hashes[name]:
            raise ValueError(f"user-log artifact missing or changed: {name}")
    questions = tuple(QuestionRecord.from_dict(row) for row in read_jsonl(
        path.parent / ARTIFACT_FILES["questions"], expected_schema_version=QUESTION_SCHEMA_VERSION,
    ))
    if len(questions) != manifest.question_count or len({q.query_id for q in questions}) != len(questions):
        raise ValueError("user-log question count or unique IDs mismatch")
    for q in questions:
        if q.query_id != _query_id(q.text, q.language, manifest.config):
            raise ValueError("user-log query identity mismatch")
        if q.language not in manifest.config.language_labels.values():
            raise ValueError("unexpected log query language")
        if q.source_targets or q.acceptable_targets or q.chat_history or q.intent_ranking:
            raise ValueError("standalone logs cannot invent anchors, history or intent predictions")
        if q.metadata.get("split") != _split(q.text, manifest.config):
            raise ValueError("user-log split mismatch")
        if q.metadata.get("data_classification") != "private":
            raise ValueError("user-log privacy declaration missing")
        if q.metadata.get("source_method") != "user_log":
            raise ValueError("user-log source method missing")
        ui_labels = q.metadata.get("ui_language_labels")
        if not isinstance(ui_labels, list) or not ui_labels or any(
            not isinstance(label, str) or manifest.config.language_labels.get(label) != q.language
            for label in ui_labels
        ):
            raise ValueError("user-log query language differs from the UI signal")
        if type(q.metadata.get("frequency")) is not int or q.metadata["frequency"] < 1:
            raise ValueError("user-log frequency must be a positive integer")
    return manifest, questions


def import_user_logs(
    csv_path: Path, config_path: Path, output_root: Path, *,
    exclusions_path: Path | None = None,
) -> Path:
    """Import without inference, source mutation, or in-place artifact updates."""
    config = TypeAdapter(UserLogConfig).validate_python(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )
    exclusions = read_json_object(exclusions_path, error_type=ValueError) if exclusions_path else {}
    if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v.strip()
           for k, v in exclusions.items()):
        raise ValueError("exclusions must map query IDs to nonempty reasons")
    source_hash = hash_file(csv_path)
    status = "imported"
    set_id = _set_id(source_hash, config, exclusions, status)
    destination = output_root / set_id.split(":", 1)[1]
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing, _ = verify_user_log_question_set(manifest_path)
        if existing.question_set_id != set_id:
            raise ValueError("existing import directory contains another question set")
        return manifest_path
    if destination.exists():
        raise ValueError("incomplete import directory exists; inspect it before retrying")

    noise = {_noise_key(t) for t in config.excluded_exact_texts}
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    seen_ids = set()
    counts: Counter[str] = Counter()
    transformations: Counter[str] = Counter()
    turn_export = isinstance(config, UserLogTurnImportConfig)
    columns = TURN_CSV_COLUMNS if turn_export else CSV_COLUMNS
    prefix = (
        re.compile(r"^" + re.escape(config.strip_leading_bot_prefix) + r"(?=$|[\s:])[\s:]*")
        if turn_export else None
    )
    # Explicit RFC-style doubled quotes. Sniffer can incorrectly infer
    # doublequote=False from the beginning of otherwise valid exports.
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=config.delimiter, doublequote=True, strict=True)
        if reader.fieldnames != list(columns):
            raise ValueError("CSV columns/order do not match the user-log export schema")
        try:
            for ordinal, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"CSV column count mismatch at record {ordinal}")
                counts["input_rows"] += 1
                text = row["text"].strip()
                prefix_removed = False
                if prefix is not None:
                    text, replacements = prefix.subn("", text, count=1)
                    prefix_removed = bool(replacements)
                    if prefix_removed:
                        transformations["leading_bot_prefix_removed"] += 1
                language = config.language_labels.get(row["language"], "")
                reason = _source_exclusion(row, config)
                if reason:
                    pass
                elif not language:
                    reason = "invalid_ui_language"
                elif not text:
                    reason = "empty_text"
                elif text.startswith("@"):
                    reason = "addressed_other_bot"
                elif len(text) == 1:
                    reason = "single_character"
                elif _noise_key(text) in noise:
                    reason = "exact_noise"
                elif config.max_characters and len(text) > config.max_characters:
                    reason = "too_long"
                query_id = _query_id(text, language, config) if text and language else None
                if query_id:
                    seen_ids.add(query_id)
                if not reason and query_id in exclusions:
                    reason = "manual_exclusion"
                counts[reason or "retained_rows"] += 1
                if not reason and prefix_removed:
                    transformations["retained_with_prefix_removed"] += 1
                groups[(normalize_log_text(text), language or row["language"], reason)].append({
                    "text": text, "source_row": ordinal, "query_id": query_id,
                    "ui_language": row["language"],
                })
        except csv.Error as exc:
            raise ValueError(f"CSV parse error near physical line {reader.line_num}") from exc
    if set(exclusions) - seen_ids:
        raise ValueError("exclusions contain query IDs absent from this CSV")
    if hash_file(csv_path) != source_hash:
        raise ValueError("source CSV changed during import")

    questions = []
    review = []
    for (_, language, reason), members in sorted(groups.items()):
        text = min(m["text"] for m in members)
        query_id = members[0]["query_id"]
        split = _split(text, config)
        flags = _flags(text, config)
        record = {
            "query_id": query_id, "text": text, "language": language,
            "ui_language_labels": sorted({m["ui_language"] for m in members}),
            "split": split, "frequency": len(members), "characters": len(text),
            "status": "excluded" if reason else "retained", "reason": reason,
            "review_flags": flags, "source_rows": [m["source_row"] for m in members],
            "exclusion_note": exclusions.get(query_id, ""),
        }
        review.append(record)
        if not reason:
            questions.append(QuestionRecord(
                query_id=query_id, text=text, language=language,
                metadata={
                    "source_method": "user_log", "data_classification": "private",
                    "split": split, "frequency": len(members),
                    "ui_language_labels": record["ui_language_labels"],
                    "review_flags": flags,
                },
            ))
    questions.sort(key=lambda q: q.query_id)
    if not questions:
        raise ValueError("no questions remain after filtering")
    report = {
        "status": status, "question_set_id": set_id, "source_hash": source_hash,
        "row_counts": dict(sorted(counts.items())), "question_count": len(questions),
        "duplicates_collapsed": counts["retained_rows"] - len(questions),
        "split_counts": dict(Counter(q.metadata["split"] for q in questions)),
        "language_counts": dict(Counter(q.language for q in questions)),
        "review_flag_counts": dict(Counter(f for q in questions for f in q.metadata["review_flags"])),
        "limitations": [
            "UI language is preserved, not detected from text.",
            "Turn 1 is an export heuristic, not a verified independent conversation.",
            "Normalized exact text groups cannot cross splits; paraphrases may.",
            "Flags are review hints, not a privacy clearance or answerability labels.",
            "Frequency is retained; existing unweighted Suite metrics count each unique query once.",
        ],
    }
    if turn_export:
        report["transformation_counts"] = dict(sorted(transformations.items()))
        report["limitations"][1] = (
            f"Export user_turn_index={config.user_turn_index} is a selection rule, "
            "not a verified independent conversation."
        )
    destination.mkdir(parents=True, mode=0o700)
    write_jsonl_atomic(destination / ARTIFACT_FILES["questions"], questions)
    write_json_object_atomic(destination / ARTIFACT_FILES["review"], {"records": review})
    write_json_object_atomic(destination / ARTIFACT_FILES["report"], report)
    manifest = UserLogQuestionSetManifest(
        question_set_id=set_id, status=status, source_hash=source_hash, config=config,
        exclusions=exclusions, question_count=len(questions),
        artifact_hashes={name: hash_file(destination / filename) for name, filename in ARTIFACT_FILES.items()},
    )
    write_json_object_atomic(manifest_path, manifest.to_dict(), validate=True)
    verify_user_log_question_set(manifest_path)
    return manifest_path
