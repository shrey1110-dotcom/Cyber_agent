"""Deterministic, duplicate-group-aware pilot balancing."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from cyber_agent.data_pipeline.config import PilotBudget
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import (
    atomic_write_json,
    atomic_write_jsonl,
    fingerprint,
    read_jsonl,
    stage_is_current,
    write_stage_marker,
)
from cyber_agent.data_pipeline.schemas import CATEGORIES, Document


PROVISIONAL_ESTIMATOR = {
    "name": "utf8_bytes_divided_by_four",
    "version": 1,
    "formula": "max(1, ceil(len(text.encode('utf-8')) / 4))",
    "exact": False,
    "limitations": "A byte heuristic used only before candidate tokenizers exist; code, Unicode, identifiers, and whitespace can differ materially.",
}


def estimate_pre_tokenizer_tokens(text: str) -> int:
    """Return a clearly provisional, deterministic token estimate."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


@dataclass(frozen=True, slots=True)
class BalanceExclusion:
    document_id: str
    duplicate_group_id: str
    source_name: str
    category: str
    estimated_tokens: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "duplicate_group_id": self.duplicate_group_id,
            "source_name": self.source_name,
            "category": self.category,
            "estimated_tokens": self.estimated_tokens,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BalanceResult:
    selected: tuple[Document, ...]
    exclusions: tuple[BalanceExclusion, ...]
    report: dict[str, Any]


def _priority(seed: int, group_id: str, target: float) -> float:
    # Use an independent deterministic domain from dataset splitting. Reusing
    # the split hash here biases every selected group toward the train range
    # whenever a balancing cap is reached.
    raw = int.from_bytes(
        hashlib.sha256(f"balance:{seed}:{group_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    return raw / max(target, 1e-12)


def balance_documents(documents: list[Document], *, budget: PilotBudget, seed: int) -> BalanceResult:
    """Apply hard source/category caps without duplicating or splitting groups."""
    groups: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        group_id = str(document.metadata.get("duplicate_group_id", document.document_id))
        groups[group_id].append(document)

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            _priority(seed, item[0], budget.category_targets[sorted(item[1], key=lambda doc: doc.document_id)[0].category]),
            item[0],
        ),
    )
    selected: list[Document] = []
    exclusions: list[BalanceExclusion] = []
    source_documents: Counter[str] = Counter()
    source_tokens: Counter[str] = Counter()
    category_tokens: Counter[str] = Counter()
    total_tokens = 0
    ended_by: Counter[str] = Counter()

    for group_id, group in ordered_groups:
        ordered = sorted(group, key=lambda document: document.document_id)
        group_tokens = {document.document_id: estimate_pre_tokenizer_tokens(document.text) for document in ordered}
        token_total = sum(group_tokens.values())
        reason: str | None = None
        if len(selected) + len(ordered) > budget.maximum_clean_documents:
            reason = "maximum_clean_documents"
        elif total_tokens + token_total > budget.maximum_estimated_tokens:
            reason = "maximum_estimated_tokens"
        else:
            for source, count in Counter(document.source_name for document in ordered).items():
                source_group_tokens = sum(group_tokens[document.document_id] for document in ordered if document.source_name == source)
                if source_documents[source] + count > budget.maximum_documents_per_source:
                    reason = "maximum_documents_per_source"
                    break
                if source_tokens[source] + source_group_tokens > budget.maximum_tokens_per_source:
                    reason = "maximum_tokens_per_source"
                    break
            if reason is None:
                for category in CATEGORIES:
                    category_group_tokens = sum(
                        group_tokens[document.document_id] for document in ordered if document.category == category
                    )
                    if category_tokens[category] + category_group_tokens > budget.maximum_tokens_per_category[category]:
                        reason = f"maximum_tokens_per_category:{category}"
                        break
        if reason is not None:
            ended_by[reason] += len(ordered)
            exclusions.extend(
                BalanceExclusion(
                    document.document_id,
                    group_id,
                    document.source_name,
                    document.category,
                    group_tokens[document.document_id],
                    reason,
                )
                for document in ordered
            )
            continue
        selected.extend(ordered)
        total_tokens += token_total
        for document in ordered:
            tokens = group_tokens[document.document_id]
            source_documents[document.source_name] += 1
            source_tokens[document.source_name] += tokens
            category_tokens[document.category] += tokens

    raw_source_documents = Counter(document.source_name for document in documents)
    raw_category_documents = Counter(document.category for document in documents)
    raw_source_tokens: Counter[str] = Counter()
    raw_category_tokens: Counter[str] = Counter()
    for document in documents:
        tokens = estimate_pre_tokenizer_tokens(document.text)
        raw_source_tokens[document.source_name] += tokens
        raw_category_tokens[document.category] += tokens
    unmet_minimums = {
        category: {
            "required": budget.minimum_tokens_per_category[category],
            "actual": category_tokens[category],
            "shortfall": budget.minimum_tokens_per_category[category] - category_tokens[category],
        }
        for category in sorted(CATEGORIES)
        if category_tokens[category] < budget.minimum_tokens_per_category[category]
    }
    concentration = {
        source: round(tokens / max(1, total_tokens) * 100.0, 6)
        for source, tokens in sorted(source_tokens.items())
    }
    report = {
        "schema_version": 1,
        "seed": seed,
        "provisional_estimator": PROVISIONAL_ESTIMATOR,
        "estimated_pre_tokenizer_tokens": total_tokens,
        "raw_documents_by_source": dict(sorted(raw_source_documents.items())),
        "final_documents_by_source": dict(sorted(source_documents.items())),
        "raw_documents_by_category": dict(sorted(raw_category_documents.items())),
        "final_documents_by_category": dict(sorted(Counter(document.category for document in selected).items())),
        "raw_tokens_by_source": dict(sorted(raw_source_tokens.items())),
        "final_tokens_by_source": dict(sorted(source_tokens.items())),
        "raw_tokens_by_category": dict(sorted(raw_category_tokens.items())),
        "final_tokens_by_category": dict(sorted(category_tokens.items())),
        "category_targets": dict(sorted(budget.category_targets.items())),
        "source_concentration_percentages": concentration,
        "unmet_minimum_tokens_per_category": unmet_minimums,
        "balancing_exclusion_count": len(exclusions),
        "balancing_exclusions_by_reason": dict(sorted(ended_by.items())),
        "collection_end_reason": "input_exhausted" if not ended_by else ended_by.most_common(1)[0][0],
        "no_document_duplication": True,
        "duplicate_groups_kept_together": True,
        "budget": budget.to_dict(),
    }
    return BalanceResult(
        tuple(sorted(selected, key=lambda document: document.document_id)),
        tuple(sorted(exclusions, key=lambda exclusion: exclusion.document_id)),
        report,
    )


def run_balance(config: PipelineConfig, *, seed: int | None = None, force: bool = False) -> dict[str, Any]:
    selected_seed = config.default_seed if seed is None else seed
    input_path = config.paths.cleaned / "deduplicated.jsonl"
    output_path = config.paths.cleaned / "balanced.jsonl"
    exclusions_path = config.paths.reports / "balance_exclusions.jsonl"
    report_path = config.paths.reports / "balance_report.json"
    input_fingerprint = fingerprint(
        [input_path, config.paths.configuration / "pilot_budget.json"],
        {"seed": selected_seed, "budget": config.pilot_budget.to_dict()},
    )
    outputs = [output_path, exclusions_path, report_path]
    if not force and stage_is_current(config.paths.manifests, "balance", input_fingerprint, outputs):
        return {"stage": "balance", "status": "skipped", "outputs": [str(path) for path in outputs]}
    documents = [Document.from_dict(value) for value in read_jsonl(input_path)]
    result = balance_documents(documents, budget=config.pilot_budget, seed=selected_seed)
    atomic_write_jsonl(output_path, (document.to_dict() for document in result.selected))
    atomic_write_jsonl(exclusions_path, (exclusion.to_dict() for exclusion in result.exclusions))
    atomic_write_json(report_path, {**result.report, "excluded_documents": [item.to_dict() for item in result.exclusions]})
    counts = {"input": len(documents), "selected": len(result.selected), "excluded": len(result.exclusions)}
    write_stage_marker(config.paths.manifests, "balance", input_fingerprint, outputs, counts)
    return {
        "stage": "balance", "status": "complete", **counts,
        "estimated_pre_tokenizer_tokens": result.report["estimated_pre_tokenizer_tokens"],
        "collection_end_reason": result.report["collection_end_reason"],
        "outputs": [str(path) for path in outputs],
    }
