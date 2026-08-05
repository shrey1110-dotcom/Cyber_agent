"""Unicode-safe normalization and the extraction/cleaning stage."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import fingerprint, iter_jsonl, stage_is_current, write_stage_marker
from cyber_agent.data_pipeline.extract import extract_text
from cyber_agent.data_pipeline.quality import assess_quality
from cyber_agent.data_pipeline.schemas import (
    Document,
    RawDocument,
    RejectionRecord,
    canonical_json,
    sha256_text,
    utc_now,
)
from cyber_agent.data_pipeline.sensitive_data import detect_sensitive_data, redact_sensitive_data


BOILERPLATE_PATTERNS = (
    re.compile(r"^(skip to (main )?content|sign in|log in|privacy policy|cookie policy)$", re.IGNORECASE),
    re.compile(r"^(home|products|pricing|documentation)(\s*[|>]\s*(home|products|pricing|documentation))*$", re.IGNORECASE),
    re.compile(r"^(copyright|all rights reserved)\b", re.IGNORECASE),
)
# reStructuredText and similar documentation formats use long runs of one
# punctuation character as heading adornments.  They are presentation syntax,
# not content; retaining them incorrectly trips the repeated-character safety
# check.  This is deliberately applied only to prose, never source code.
STRUCTURAL_ADORNMENT = re.compile(r"^[!#%&'*+,-./:=?@^_`|~]{4,}$")


def remove_control_characters(value: str) -> str:
    return "".join(
        character
        for character in value
        if character in {"\n", "\t"} or unicodedata.category(character) not in {"Cc", "Cf", "Cs"}
    )


def normalize_text(value: str, *, preserve_code: bool) -> str:
    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = remove_control_characters(normalized)
    if preserve_code:
        lines = [line.rstrip() for line in normalized.split("\n")]
        return "\n".join(lines).strip("\n")

    kept_lines: list[str] = []
    seen_short_lines: set[str] = set()
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line and STRUCTURAL_ADORNMENT.fullmatch(line):
            continue
        if line and any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
            continue
        comparison = line.casefold()
        if line and len(line) <= 100 and comparison in seen_short_lines:
            continue
        if line and len(line) <= 100:
            seen_short_lines.add(comparison)
        kept_lines.append(line)
    joined = "\n".join(kept_lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def run_clean(config: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    raw_path = config.paths.raw / "documents.jsonl"
    extracted_path = config.paths.extracted / "documents.jsonl"
    cleaned_path = config.paths.cleaned / "documents.jsonl"
    rejected_path = config.paths.rejected / "clean.jsonl"
    input_fingerprint = fingerprint([raw_path], config.fingerprint_payload())
    outputs = [extracted_path, cleaned_path, rejected_path]
    if not force and stage_is_current(config.paths.manifests, "clean", input_fingerprint, outputs):
        return {"stage": "clean", "status": "skipped", "outputs": [str(path) for path in outputs]}

    # Keep the clean stage bounded: the research collection can exceed the RAM
    # budget of a laptop even though individual documents are deliberately small.
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=".clean.", suffix=".tmp", dir=config.paths.data))
    extracted_temporary = temporary / "extracted.jsonl"
    cleaned_temporary = temporary / "cleaned.jsonl"
    rejected_temporary = temporary / "rejected.jsonl"
    input_count = 0
    accepted_count = 0
    rejected_count = 0
    try:
        assert temporary is not None
        with (
            extracted_temporary.open("x", encoding="utf-8", newline="\n") as extracted_handle,
            cleaned_temporary.open("x", encoding="utf-8", newline="\n") as cleaned_handle,
            rejected_temporary.open("x", encoding="utf-8", newline="\n") as rejected_handle,
        ):
            for value in iter_jsonl(raw_path):
                input_count += 1
                raw = RawDocument.from_dict(value)
                try:
                    extracted_text, extraction_metadata = extract_text(raw.raw_text, raw.media_type)
                except ValueError as exc:
                    rejected_handle.write(
                        canonical_json(_rejection(raw, "extract", "extraction_failed", str(exc)).to_dict()) + "\n"
                    )
                    rejected_count += 1
                    continue

                normalized = normalize_text(extracted_text, preserve_code=raw.category == "code")
                findings = detect_sensitive_data(normalized)
                if findings and config.sensitive_data_action == "reject":
                    rejection = _rejection(
                        raw,
                        "sensitive_data",
                        "sensitive_data_detected",
                        "sensitive or personal data detected",
                        tuple(sorted({finding.kind for finding in findings})),
                    )
                    rejected_handle.write(canonical_json(rejection.to_dict()) + "\n")
                    rejected_count += 1
                    continue
                if findings:
                    normalized = redact_sensitive_data(normalized, findings)

                extracted = RawDocument(
                    document_id=raw.document_id,
                    raw_text=normalized,
                    source_name=raw.source_name,
                    source_url=raw.source_url,
                    license=raw.license,
                    category=raw.category,
                    language=raw.language,
                    retrieved_at=raw.retrieved_at,
                    media_type="text/plain",
                    attribution_requirements=raw.attribution_requirements,
                    metadata={**raw.metadata, **extraction_metadata},
                )
                extracted_handle.write(canonical_json(extracted.to_dict()) + "\n")

                assessment = assess_quality(normalized, raw.category, config)
                if not assessment.accepted:
                    rejection = _rejection(
                        raw,
                        "quality",
                        "quality_rejected",
                        assessment.summary,
                        tuple(assessment.reason_codes),
                    )
                    rejected_handle.write(canonical_json(rejection.to_dict()) + "\n")
                    rejected_count += 1
                    continue
                metadata = {
                    **extracted.metadata,
                    "attribution_requirements": raw.attribution_requirements,
                    "quality_components": assessment.components,
                }
                cleaned = Document(
                    document_id=raw.document_id,
                    text=normalized,
                    source_name=raw.source_name,
                    source_url=raw.source_url,
                    license=raw.license,
                    category=raw.category,
                    language=raw.language,
                    retrieved_at=raw.retrieved_at,
                    content_hash=sha256_text(normalized),
                    quality_score=assessment.score,
                    metadata=metadata,
                )
                cleaned_handle.write(canonical_json(cleaned.to_dict()) + "\n")
                accepted_count += 1
            for handle in (extracted_handle, cleaned_handle, rejected_handle):
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(extracted_temporary, extracted_path)
        os.replace(cleaned_temporary, cleaned_path)
        os.replace(rejected_temporary, rejected_path)
        shutil.rmtree(temporary)
        temporary = None
        counts = {"input": input_count, "accepted": accepted_count, "rejected": rejected_count}
        write_stage_marker(config.paths.manifests, "clean", input_fingerprint, outputs, counts)
        return {"stage": "clean", "status": "complete", **counts, "outputs": [str(path) for path in outputs]}
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _rejection(
    raw: RawDocument,
    stage: str,
    primary_code: str,
    reason: str,
    detail_codes: tuple[str, ...] = (),
) -> RejectionRecord:
    return RejectionRecord(
        document_id=raw.document_id,
        source_name=raw.source_name,
        stage=stage,  # type: ignore[arg-type]
        reason=reason,
        reason_codes=(primary_code, *detail_codes),
        rejected_at=utc_now(),
        metadata={"source_url_hash": sha256_text(raw.source_url)},
    )
