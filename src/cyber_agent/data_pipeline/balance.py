"""Deterministic, duplicate-group-aware pilot balancing."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import PilotBudget
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import (
    atomic_write_json,
    atomic_write_jsonl,
    fingerprint,
    iter_jsonl,
    read_jsonl,
    stage_is_current,
    write_stage_marker,
)
from cyber_agent.data_pipeline.schemas import CATEGORIES, Document, canonical_json


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
    # Stream large collections. Deduplication has already reduced each group
    # to a keeper, while this implementation still remembers group decisions
    # so a repeated group cannot be split by a cap.
    temporary_dir = Path(tempfile.mkdtemp(prefix=".balance.", suffix=".tmp", dir=config.paths.data))
    temporary_output = temporary_dir / output_path.name
    temporary_exclusions = temporary_dir / exclusions_path.name
    counts = {"input": 0, "selected": 0, "excluded": 0}
    source_documents: Counter[str] = Counter(); source_tokens: Counter[str] = Counter()
    category_tokens: Counter[str] = Counter(); category_documents: Counter[str] = Counter(); raw_source_documents: Counter[str] = Counter()
    raw_category_documents: Counter[str] = Counter(); raw_source_tokens: Counter[str] = Counter(); raw_category_tokens: Counter[str] = Counter()
    ended_by: Counter[str] = Counter(); selected_tokens = 0; group_decisions: dict[str, bool] = {}
    budget = config.pilot_budget
    try:
        with temporary_output.open("x", encoding="utf-8", newline="\n") as kept, temporary_exclusions.open("x", encoding="utf-8", newline="\n") as excluded:
            for value in iter_jsonl(input_path):
                document = Document.from_dict(value); counts["input"] += 1
                raw_source_documents[document.source_name] += 1; raw_category_documents[document.category] += 1
                estimate = estimate_pre_tokenizer_tokens(document.text)
                raw_source_tokens[document.source_name] += estimate; raw_category_tokens[document.category] += estimate
                group_id = str(document.metadata.get("duplicate_group_id", document.document_id))
                decision = group_decisions.get(group_id)
                reason: str | None = None
                if decision is None:
                    if counts["selected"] >= budget.maximum_clean_documents: reason = "maximum_clean_documents"
                    elif selected_tokens + estimate > budget.maximum_estimated_tokens: reason = "maximum_estimated_tokens"
                    elif source_documents[document.source_name] + 1 > budget.maximum_documents_per_source: reason = "maximum_documents_per_source"
                    elif source_tokens[document.source_name] + estimate > budget.maximum_tokens_per_source: reason = "maximum_tokens_per_source"
                    elif category_tokens[document.category] + estimate > budget.maximum_tokens_per_category[document.category]: reason = f"maximum_tokens_per_category:{document.category}"
                    decision = reason is None; group_decisions[group_id] = decision
                if decision:
                    kept.write(canonical_json(document.to_dict()) + "\n"); counts["selected"] += 1; selected_tokens += estimate
                    source_documents[document.source_name] += 1; source_tokens[document.source_name] += estimate; category_tokens[document.category] += estimate; category_documents[document.category] += 1
                else:
                    reason = reason or "duplicate_group_already_excluded"; counts["excluded"] += 1; ended_by[reason] += 1
                    excluded.write(canonical_json(BalanceExclusion(document.document_id, group_id, document.source_name, document.category, estimate, reason).to_dict()) + "\n")
            kept.flush(); excluded.flush(); os.fsync(kept.fileno()); os.fsync(excluded.fileno())
        report = {
            "schema_version": 1, "seed": selected_seed, "provisional_estimator": PROVISIONAL_ESTIMATOR,
            "estimated_pre_tokenizer_tokens": selected_tokens,
            "raw_documents_by_source": dict(sorted(raw_source_documents.items())), "final_documents_by_source": dict(sorted(source_documents.items())),
            "raw_documents_by_category": dict(sorted(raw_category_documents.items())), "final_documents_by_category": dict(sorted(category_documents.items())),
            "raw_tokens_by_source": dict(sorted(raw_source_tokens.items())), "final_tokens_by_source": dict(sorted(source_tokens.items())),
            "raw_tokens_by_category": dict(sorted(raw_category_tokens.items())), "final_tokens_by_category": dict(sorted(category_tokens.items())),
            "category_targets": dict(sorted(budget.category_targets.items())),
            "source_concentration_percentages": {s: round(t / max(1, selected_tokens) * 100.0, 6) for s, t in sorted(source_tokens.items())},
            "unmet_minimum_tokens_per_category": {c: {"required": budget.minimum_tokens_per_category[c], "actual": category_tokens[c], "shortfall": budget.minimum_tokens_per_category[c] - category_tokens[c]} for c in sorted(CATEGORIES) if category_tokens[c] < budget.minimum_tokens_per_category[c]},
            "balancing_exclusion_count": counts["excluded"], "balancing_exclusions_by_reason": dict(sorted(ended_by.items())),
            "collection_end_reason": "input_exhausted" if not ended_by else ended_by.most_common(1)[0][0], "no_document_duplication": True, "duplicate_groups_kept_together": True, "budget": budget.to_dict(),
        }
        os.replace(temporary_output, output_path); os.replace(temporary_exclusions, exclusions_path)
        atomic_write_json(report_path, report)
    finally:
        import shutil; shutil.rmtree(temporary_dir, ignore_errors=True)
    write_stage_marker(config.paths.manifests, "balance", input_fingerprint, outputs, counts)
    return {
        "stage": "balance", "status": "complete", **counts,
        "estimated_pre_tokenizer_tokens": report["estimated_pre_tokenizer_tokens"],
        "collection_end_reason": report["collection_end_reason"],
        "outputs": [str(path) for path in outputs],
    }
