"""Documented, deterministic quality scoring and rejection rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.schemas import Category


ENGLISH_MARKERS = frozenset(
    {"a", "an", "and", "are", "as", "be", "before", "by", "can", "for", "from", "in", "is", "it", "of", "on", "or", "should", "that", "the", "this", "to", "with"}
)
NAVIGATION_WORDS = frozenset({"home", "menu", "pricing", "products", "login", "signin", "subscribe", "contact", "next", "previous"})
SPAM_PHRASES = ("buy now", "click here", "free money", "online casino", "limited time offer")


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    accepted: bool
    score: float
    reason_codes: tuple[str, ...]
    summary: str
    components: dict[str, float]


def assess_quality(text: str, category: Category, config: PipelineConfig) -> QualityAssessment:
    reasons: list[str] = []
    if not text.strip():
        reasons.append("empty")
    length = len(text)
    if length < config.minimum_document_characters:
        reasons.append("too_short")
    if length > config.maximum_document_characters:
        reasons.append("too_long")
    if text.count("\ufffd") / max(1, length) > 0.01:
        reasons.append("corrupted_text")

    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", text.casefold())
    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    if len(tokens) >= 20 and unique_ratio < 0.18:
        reasons.append("pathological_repetition")
    repeated_character = bool(re.search(r"(.)\1{15,}", text, re.DOTALL))
    if repeated_character:
        reasons.append("repeated_characters")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.35:
        reasons.append("repeated_lines")

    markup_ratio = sum(len(match.group(0)) for match in re.finditer(r"<[^>]{1,200}>", text)) / max(1, length)
    if markup_ratio > 0.35:
        reasons.append("mostly_markup")
    navigation_ratio = sum(token in NAVIGATION_WORDS for token in tokens) / max(1, len(tokens))
    if navigation_ratio > 0.30:
        reasons.append("mostly_navigation")
    if sum(text.casefold().count(phrase) for phrase in SPAM_PHRASES) >= 2:
        reasons.append("obvious_spam")

    printable_ratio = sum(character.isprintable() or character in "\n\t" for character in text) / max(1, length)
    if printable_ratio < 0.95:
        reasons.append("unreadable")
    english_ratio = sum(token in ENGLISH_MARKERS for token in tokens) / max(1, len(tokens))
    if category != "code" and len(tokens) >= 20 and english_ratio < 0.045:
        reasons.append("not_english")
    if len(tokens) < 8 and category != "code":
        reasons.append("low_information")

    components = {
        "length": min(1.0, length / 500.0),
        "diversity": min(1.0, unique_ratio / 0.60),
        "readability": printable_ratio,
        "english": 1.0 if category == "code" else min(1.0, english_ratio / 0.12),
        "structure": 1.0 - min(1.0, markup_ratio + navigation_ratio),
    }
    score = round(
        0.20 * components["length"]
        + 0.20 * components["diversity"]
        + 0.20 * components["readability"]
        + 0.20 * components["english"]
        + 0.20 * components["structure"],
        6,
    )
    hard_reasons = tuple(dict.fromkeys(reasons))
    if score < config.minimum_quality_score:
        hard_reasons = (*hard_reasons, "quality_score_below_threshold")
    accepted = not hard_reasons
    summary = "accepted" if accepted else ", ".join(hard_reasons)
    return QualityAssessment(accepted, score, hard_reasons, summary, components)

