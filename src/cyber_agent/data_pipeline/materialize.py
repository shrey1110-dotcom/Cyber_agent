"""Convert safely downloaded, non-executable source files into local manifests."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from cyber_agent.data_pipeline.balance import estimate_pre_tokenizer_tokens
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import atomic_write_json, atomic_write_jsonl
from cyber_agent.data_pipeline.sources import SourceDefinition


SKIP_PATH_PARTS = frozenset({
    ".git", "vendor", "vendors", "vendored", "node_modules", "dist", "build",
    "generated", "__pycache__", ".venv", "venv", "third_party", "third-party",
})
CODE_EXTENSIONS = frozenset({".py", ".sh", ".bash", ".json", ".yaml", ".yml", ".dockerfile"})


def _safe_text(path: Path, maximum_bytes: int) -> str | None:
    if path.stat().st_size < 80 or path.stat().st_size > maximum_bytes:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None
    lowered_name = path.name.casefold()
    if ".min." in lowered_name or lowered_name.endswith((".map", ".lock")):
        return None
    lines = text.splitlines()
    if not lines or max((len(line) for line in lines), default=0) > 20_000:
        return None
    if len(lines) < 3 and len(text) > 2_000:
        return None
    return text


def _category(source: SourceDefinition, path: Path, text: str) -> str:
    suffix = path.suffix.casefold()
    name = path.name.casefold()
    searchable = f"{path.as_posix()} {text[:2000]}".casefold()
    if suffix in CODE_EXTENSIONS or name in {"dockerfile", "containerfile"}:
        return "code"
    if any(term in searchable for term in ("cve-", "cwe-", "attack technique", "vulnerability", "threat actor")):
        return "cybersecurity"
    if any(term in searchable for term in ("ipv4", "ipv6", "socket", "tcp", "udp", "dns", "network interface")):
        return "networking"
    if any(term in searchable for term in ("linux", "systemd", "filesystem", "process", "daemon", "permission")):
        return "linux"
    return source.category


def _language(path: Path) -> str:
    suffix = path.suffix.casefold()
    return {
        ".py": "python", ".sh": "bash", ".bash": "bash", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".dockerfile": "dockerfile",
    }.get(suffix, "en")


def _selected_paths(root: Path, source: SourceDefinition) -> Iterable[Path]:
    options = source.adapter_options or {}
    extensions = {str(item).casefold() for item in options.get("extensions", [])}
    prefixes = tuple(str(item) for item in options.get("path_prefixes", []))
    contains = tuple(str(item) for item in options.get("path_contains", []))
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or any(part.casefold() in SKIP_PATH_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.casefold()
        if extensions and suffix not in extensions and path.name.casefold() not in {"dockerfile", "containerfile"}:
            continue
        if prefixes and not relative.startswith(prefixes):
            continue
        if contains and not any(fragment in f"/{relative}" for fragment in contains):
            continue
        yield path


def _write_records(
    config: PipelineConfig,
    source: SourceDefinition,
    records: Iterable[tuple[str, str, str, dict[str, Any]]],
    *,
    token_limit: int | None = None,
) -> dict[str, Any]:
    root = config.paths.project_root / source.data_location
    root = root.parent
    if root.exists():
        manifest = root / "manifest.jsonl"
        if manifest.exists():
            report_path = root / "materialization_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            return {
                "status": "skipped", "manifest": str(manifest), "reason": "already_materialized",
                "documents": int(report.get("documents", 0)),
                "estimated_pre_tokenizer_tokens": int(report.get("estimated_pre_tokenizer_tokens", 0)),
                "collection_end_reason": report.get("collection_end_reason", "input_exhausted"),
            }
        raise ValueError(f"source materialization directory is incomplete: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{root.name}.", suffix=".tmp", dir=root.parent))
    count = 0
    estimated_tokens = 0
    stop_reason = "input_exhausted"
    manifest_rows: list[dict[str, Any]] = []
    maximum_tokens = min(
        config.pilot_budget.maximum_tokens_per_source,
        token_limit if token_limit is not None else config.pilot_budget.maximum_tokens_per_source,
    )
    try:
        assert temporary is not None
        documents = temporary / "documents"
        documents.mkdir()
        for source_identifier, text, category, metadata in records:
            tokens = estimate_pre_tokenizer_tokens(text)
            if count >= config.pilot_budget.maximum_documents_per_source:
                stop_reason = "maximum_documents_per_source"
                break
            if estimated_tokens + tokens > maximum_tokens:
                stop_reason = "target_estimated_tokens" if token_limit is not None and maximum_tokens == token_limit else "maximum_tokens_per_source"
                break
            suffix = ".json" if metadata.get("language") == "json" else ".txt"
            relative = Path("documents") / f"document-{count:05d}{suffix}"
            (temporary / relative).write_text(text.rstrip() + "\n", encoding="utf-8")
            record_metadata = {
                **metadata,
                "original_identifier": source_identifier,
                "stated_license_or_terms": source.license,
                "local_research_only": True,
            }
            if category == "code":
                record_metadata.setdefault("repository", (source.adapter_options or {}).get("repository", source.homepage))
                record_metadata.setdefault("revision", (source.adapter_options or {}).get("revision", source.exact_release_or_version))
                record_metadata.setdefault("detected_licenses", [source.license])
            manifest_rows.append({
                "path": relative.as_posix(),
                "source_url": f"{source.download_location}#{source_identifier}",
                "source_release": source.exact_release_or_version,
                "license": source.license,
                "category": category,
                "language": "en",
                "retrieved_at": source.retrieved_at,
                "media_type": metadata.get("media_type", "text/plain"),
                "metadata": record_metadata,
            })
            count += 1
            estimated_tokens += tokens
        atomic_write_jsonl(temporary / "manifest.jsonl", manifest_rows)
        atomic_write_json(temporary / "materialization_report.json", {
            "schema_version": 1,
            "source_name": source.source_name,
            "exact_release_or_version": source.exact_release_or_version,
            "documents": count,
            "estimated_pre_tokenizer_tokens": estimated_tokens,
            "collection_end_reason": stop_reason,
            "local_research_only": True,
            "downloaded_code_executed": False,
        })
        os.replace(temporary, root)
        temporary = None
        return {
            "status": "complete", "manifest": str(root / "manifest.jsonl"),
            "documents": count, "estimated_pre_tokenizer_tokens": estimated_tokens,
            "collection_end_reason": stop_reason,
        }
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def materialize_archive(
    config: PipelineConfig,
    source: SourceDefinition,
    extracted_root: Path,
    *,
    token_limit: int | None = None,
) -> dict[str, Any]:
    maximum_file_bytes = min(config.maximum_document_characters * 4, 2_000_000)

    def records() -> Iterable[tuple[str, str, str, dict[str, Any]]]:
        for path in _selected_paths(extracted_root, source):
            text = _safe_text(path, maximum_file_bytes)
            if text is None:
                continue
            relative = path.relative_to(extracted_root)
            language = _language(path)
            yield relative.as_posix(), text, _category(source, relative, text), {
                "programming_language": language,
                "original_path": relative.as_posix(),
                "repository": (source.adapter_options or {}).get("repository", source.homepage),
                "revision": (source.adapter_options or {}).get("revision", source.exact_release_or_version),
            }

    return _write_records(config, source, records(), token_limit=token_limit)


def materialize_stix(
    config: PipelineConfig,
    source: SourceDefinition,
    downloaded: Path,
    *,
    token_limit: int | None = None,
) -> dict[str, Any]:
    payload = json.loads(downloaded.read_text(encoding="utf-8"))
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("STIX source does not contain an objects array")

    def records() -> Iterable[tuple[str, str, str, dict[str, Any]]]:
        for item in objects:
            if not isinstance(item, dict) or item.get("revoked") or item.get("x_mitre_deprecated"):
                continue
            identifier = str(item.get("id", ""))
            name = str(item.get("name", ""))
            description = str(item.get("description", ""))
            external_ids = [
                reference.get("external_id") for reference in item.get("external_references", [])
                if isinstance(reference, dict) and reference.get("external_id")
            ]
            if len(description) < 80:
                continue
            text = json.dumps({
                "type": item.get("type"), "id": identifier, "name": name,
                "external_ids": external_ids, "description": description,
                "kill_chain_phases": item.get("kill_chain_phases", []),
                "platforms": item.get("x_mitre_platforms", []),
                "detection": item.get("x_mitre_detection", ""),
            }, ensure_ascii=False, indent=2, sort_keys=True)
            yield identifier, text, "cybersecurity", {"programming_language": "json", "media_type": "application/json", "stix_type": item.get("type")}

    return _write_records(config, source, records(), token_limit=token_limit)


def _xml_text(element: ET.Element) -> str:
    text = " ".join(part.strip() for part in element.itertext() if part.strip())
    return re.sub(r"\s+", " ", text).strip()


def materialize_cwe(
    config: PipelineConfig,
    source: SourceDefinition,
    xml_path: Path,
    *,
    token_limit: int | None = None,
) -> dict[str, Any]:
    def records() -> Iterable[tuple[str, str, str, dict[str, Any]]]:
        for _, element in ET.iterparse(xml_path, events=("end",)):
            if not element.tag.endswith("Weakness"):
                continue
            identifier = element.attrib.get("ID", "unknown")
            name = element.attrib.get("Name", "Unnamed weakness")
            body = _xml_text(element)
            if len(body) >= 80:
                yield f"CWE-{identifier}", f"CWE-{identifier}: {name}\n\n{body}", "cybersecurity", {"programming_language": "not_applicable", "cwe_id": f"CWE-{identifier}"}
            element.clear()

    return _write_records(config, source, records(), token_limit=token_limit)
