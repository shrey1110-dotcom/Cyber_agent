"""Convert safely downloaded, non-executable source files into local manifests."""

from __future__ import annotations

import bz2
import html
import json
import os
import re
import shutil
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

from cyber_agent.data_pipeline.balance import estimate_pre_tokenizer_tokens
from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.extract import html_to_text
from cyber_agent.data_pipeline.export import atomic_write_json
from cyber_agent.data_pipeline.schemas import canonical_json
from cyber_agent.data_pipeline.sources import SourceDefinition


SKIP_PATH_PARTS = frozenset({
    ".git", "vendor", "vendors", "vendored", "node_modules", "dist", "build",
    "generated", "__pycache__", ".venv", "venv", "third_party", "third-party",
})
CODE_EXTENSIONS = frozenset({
    ".py", ".sh", ".bash", ".go", ".rs", ".c", ".h", ".cc", ".cpp",
    ".ps1", ".json", ".yaml", ".yml", ".dockerfile",
})
WIKITEXT_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
WIKITEXT_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
WIKITEXT_REFERENCE = re.compile(r"<ref(?:\s[^>]*)?>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
WIKITEXT_TAG = re.compile(
    r"</?(?:ref|references|gallery|timeline|math|score|syntaxhighlight|source|pre|nowiki)(?:\s[^>]*)?>",
    re.IGNORECASE,
)
WIKITEXT_LINK_WITH_LABEL = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")
WIKITEXT_LINK = re.compile(r"\[\[([^\]]+)\]\]")
WIKITEXT_EXTERNAL_LINK = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
WIKITEXT_HEADING = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", re.MULTILINE)


def _wikitext_to_plain_text(value: str) -> str:
    """Apply a deliberately conservative, non-executing MediaWiki cleanup.

    It is not a renderer.  It only strips common structural syntax before the
    normal Phase 2 text/quality pipeline runs, preserving article prose and
    source attribution while avoiding template-heavy navigation text.
    """
    text = WIKITEXT_COMMENT.sub("", value)
    text = WIKITEXT_REFERENCE.sub("", text)
    # Repeatedly remove innermost templates; a bound prevents adversarially
    # deep markup from consuming unbounded CPU.
    for _ in range(64):
        updated = WIKITEXT_TEMPLATE.sub("", text)
        if updated == text:
            break
        text = updated
    text = WIKITEXT_TAG.sub("", text)
    text = WIKITEXT_EXTERNAL_LINK.sub(r"\1", text)
    text = WIKITEXT_LINK_WITH_LABEL.sub(r"\1", text)
    text = WIKITEXT_LINK.sub(r"\1", text)
    text = WIKITEXT_HEADING.sub(r"\1", text)
    text = re.sub(r"(?m)^\s*__[A-Z_]+__\s*$", "", text)
    text = re.sub(r"(?m)^\s*[*#:;]+\s*", "", text)
    text = re.sub(r"(?m)^\s*\|[-+}]?.*$", "", text)
    return html.unescape(text).strip()


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
        ".py": "python", ".sh": "bash", ".bash": "bash", ".go": "go",
        ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
        ".ps1": "powershell", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".md": "markdown", ".rst": "restructuredtext", ".adoc": "asciidoc",
        ".dockerfile": "dockerfile",
    }.get(suffix, "en")


def _selected_paths(root: Path, source: SourceDefinition) -> Iterable[Path]:
    options = source.adapter_options or {}
    extensions = {str(item).casefold() for item in options.get("extensions", [])}
    prefixes = tuple(str(item) for item in options.get("path_prefixes", []))
    contains = tuple(str(item) for item in options.get("path_contains", []))
    allowed_path_parts = {str(item).casefold() for item in options.get("allow_path_parts", [])}
    unknown_overrides = allowed_path_parts - SKIP_PATH_PARTS
    if unknown_overrides:
        raise ValueError(f"source path override is not a normally skipped path part: {sorted(unknown_overrides)}")
    blocked_parts = SKIP_PATH_PARTS - allowed_path_parts
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or any(part.casefold() in blocked_parts for part in path.parts):
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
    maximum_tokens = min(
        config.pilot_budget.maximum_tokens_per_source,
        token_limit if token_limit is not None else config.pilot_budget.maximum_tokens_per_source,
    )
    try:
        assert temporary is not None
        documents = temporary / "documents"
        documents.mkdir()
        # The temporary directory itself is published atomically below.  Writing
        # the manifest progressively avoids retaining hundreds of thousands of
        # source records in memory during larger, bounded research collections.
        with (temporary / "manifest.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as manifest:
            for source_identifier, text, category, metadata in records:
                tokens = estimate_pre_tokenizer_tokens(text)
                if count >= config.pilot_budget.maximum_documents_per_source:
                    stop_reason = "maximum_documents_per_source"
                    break
                if estimated_tokens + tokens > maximum_tokens:
                    stop_reason = (
                        "target_estimated_tokens"
                        if token_limit is not None and maximum_tokens == token_limit
                        else "maximum_tokens_per_source"
                    )
                    break
                suffix = ".json" if metadata.get("language") == "json" else ".txt"
                relative = Path("documents") / f"document-{count:05d}{suffix}"
                (temporary / relative).write_text(text.rstrip() + "\n", encoding="utf-8")
                source_url = metadata.get("source_url")
                if not isinstance(source_url, str) or not source_url.strip():
                    source_url = f"{source.download_location}#{source_identifier}"
                record_license = metadata.get("record_license", source.license)
                if not isinstance(record_license, str) or not record_license.strip():
                    raise ValueError("materialized record is missing an explicit license")
                record_metadata = {
                    **metadata,
                    "record_license": record_license,
                    "original_identifier": source_identifier,
                    "stated_license_or_terms": source.license,
                    "local_research_only": True,
                }
                if category == "code":
                    record_metadata.setdefault(
                        "repository", (source.adapter_options or {}).get("repository", source.homepage)
                    )
                    record_metadata.setdefault(
                        "revision",
                        (source.adapter_options or {}).get("revision", source.exact_release_or_version),
                    )
                    record_metadata.setdefault("detected_licenses", [source.license])
                manifest.write(
                    canonical_json(
                        {
                            "path": relative.as_posix(),
                            "source_url": source_url,
                            "source_release": source.exact_release_or_version,
                            "license": record_license,
                            "category": category,
                            "language": "en",
                            "retrieved_at": source.retrieved_at,
                            "media_type": metadata.get("media_type", "text/plain"),
                            "metadata": record_metadata,
                        }
                    )
                    + "\n"
                )
                count += 1
                estimated_tokens += tokens
            manifest.flush()
            os.fsync(manifest.fileno())
        if count == 0:
            raise ValueError(
                "source materialization produced no eligible records; "
                "check the pinned release and configured path/extension filters"
            )
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


def _stackexchange_license_for_creation_date(value: str) -> str:
    """Map the published Stack Exchange post-license eras without guessing.

    The archive is a multi-license source: the precise post creation time
    determines the retained CC BY-SA version.  A malformed or missing timestamp
    is rejected rather than defaulting to the source's newest license.
    """
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        date = parsed.date().isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("Stack Exchange post has an invalid CreationDate") from exc
    if date < "2011-04-08":
        return "CC-BY-SA-2.5"
    if date < "2018-05-02":
        return "CC-BY-SA-3.0"
    return "CC-BY-SA-4.0"


def _stackexchange_category(tags: str, source: SourceDefinition) -> str:
    normalized = tags.casefold()
    if any(term in normalized for term in ("security", "cryptography", "vulnerability", "malware", "authentication")):
        return "cybersecurity"
    if any(term in normalized for term in ("network", "http", "tcp", "udp", "dns", "socket", "ipv6", "ipv4")):
        return "networking"
    if any(term in normalized for term in ("linux", "bash", "shell", "command-line", "unix")):
        return "linux"
    return source.category


def materialize_stackexchange_posts(
    config: PipelineConfig,
    source: SourceDefinition,
    extracted_xml: Path,
    *,
    token_limit: int | None = None,
) -> dict[str, Any]:
    """Stream one reviewed Stack Exchange Posts.xml file into attributed records.

    The function never executes snippets or markup.  It retains stable post
    identifiers, canonical permalinks, post type, creation time, site identity,
    and the per-post CC BY-SA license era.  The normal Phase 2 sensitive-data,
    quality, and deduplication gates remain mandatory after materialization.
    """
    if source.license != "MULTIPLE-SPDX-REQUIRED":
        raise ValueError("Stack Exchange source must declare MULTIPLE-SPDX-REQUIRED")
    if extracted_xml.stat().st_size > config.pilot_budget.maximum_decompressed_bytes:
        raise ValueError("Stack Exchange XML exceeds configured decompressed-byte limit")
    options = source.adapter_options or {}
    site_base_url = options.get("site_base_url")
    if not isinstance(site_base_url, str) or not site_base_url.startswith("https://"):
        raise ValueError("Stack Exchange source requires an HTTPS site_base_url")

    def records() -> Iterable[tuple[str, str, str, dict[str, Any]]]:
        consumed = 0
        with extracted_xml.open("rb") as stream:
            head = stream.read(64 * 1024)
            if b"<!DOCTYPE" in head.upper() or b"<!ENTITY" in head.upper():
                raise ValueError("Stack Exchange XML must not contain DTD or entity declarations")

            class _BoundedReader:
                def __init__(self, prefix: bytes) -> None:
                    self._prefix = prefix

                def read(self, size: int = -1) -> bytes:
                    nonlocal consumed
                    if self._prefix:
                        if size < 0 or size >= len(self._prefix):
                            chunk, self._prefix = self._prefix, b""
                        else:
                            chunk, self._prefix = self._prefix[:size], self._prefix[size:]
                    else:
                        chunk = stream.read(size)
                    consumed += len(chunk)
                    if consumed > config.pilot_budget.maximum_decompressed_bytes:
                        raise ValueError("Stack Exchange XML exceeds configured decompressed-byte limit")
                    return chunk

            for _event, element in ET.iterparse(_BoundedReader(head), events=("end",)):
                if element.tag.rsplit("}", 1)[-1] != "row":
                    continue
                attributes = element.attrib
                try:
                    post_id = attributes["Id"]
                    post_type = attributes["PostTypeId"]
                    created_at = attributes["CreationDate"]
                    body = attributes["Body"]
                except KeyError:
                    element.clear()
                    continue
                if post_type not in {"1", "2"}:
                    element.clear()
                    continue
                record_license = _stackexchange_license_for_creation_date(created_at)
                title = html_to_text(html.unescape(attributes.get("Title", ""))).strip()
                tags = html.unescape(attributes.get("Tags", ""))
                body_text = html_to_text(html.unescape(body)).strip()
                if not body_text:
                    element.clear()
                    continue
                heading = "Question" if post_type == "1" else "Answer"
                text = f"{heading}\nTitle: {title}\nTags: {tags}\n\n{body_text}".strip()
                permalink = f"{site_base_url}/questions/{post_id}" if post_type == "1" else f"{site_base_url}/a/{post_id}"
                category = _stackexchange_category(tags, source)
                metadata: dict[str, Any] = {
                    "source_url": permalink,
                    "media_type": "text/plain",
                    "record_license": record_license,
                    "license_period": record_license,
                    "stackexchange_site": site_base_url,
                    "post_id": post_id,
                    "post_type": "question" if post_type == "1" else "answer",
                    "parent_id": attributes.get("ParentId", ""),
                    "post_creation_date": created_at,
                    "post_tags": tags,
                    "owner_user_id": attributes.get("OwnerUserId", ""),
                    "last_editor_user_id": attributes.get("LastEditorUserId", ""),
                }
                if category == "code":
                    metadata.update({
                        "code_provenance_kind": "stackexchange_post",
                        "detected_licenses": [record_license],
                    })
                yield post_id, text, category, metadata
                element.clear()

    return _write_records(config, source, records(), token_limit=token_limit)


def materialize_wikimedia_xml_bz2(
    config: PipelineConfig,
    source: SourceDefinition,
    downloaded: Path,
    *,
    token_limit: int | None = None,
) -> dict[str, Any]:
    """Stream a reviewed Wikimedia articles dump without extracting an archive.

    The source configuration provides an exact dated dump URL, one declared
    license, and attribution instructions.  The reader accepts main-namespace
    pages only, rejects a DTD/entity declaration before parsing, accounts for
    decompressed bytes, and never evaluates templates, Lua modules, or scripts.
    """
    maximum_bytes = config.pilot_budget.maximum_decompressed_bytes

    def records() -> Iterable[tuple[str, str, str, dict[str, Any]]]:
        decompressed = 0
        with bz2.open(downloaded, "rb") as stream:
            head = stream.read(64 * 1024)
            if b"<!DOCTYPE" in head.upper() or b"<!ENTITY" in head.upper():
                raise ValueError("Wikimedia XML must not contain DTD or entity declarations")

            class _BoundedReader:
                def __init__(self, prefix: bytes) -> None:
                    self._prefix = prefix

                def read(self, size: int = -1) -> bytes:
                    nonlocal decompressed
                    if self._prefix:
                        if size < 0 or size >= len(self._prefix):
                            chunk, self._prefix = self._prefix, b""
                        else:
                            chunk, self._prefix = self._prefix[:size], self._prefix[size:]
                    else:
                        chunk = stream.read(size)
                    decompressed += len(chunk)
                    if decompressed > maximum_bytes:
                        raise ValueError("Wikimedia dump exceeds configured decompressed-byte limit")
                    return chunk

            parser = ET.iterparse(_BoundedReader(head), events=("end",))
            for _event, page in parser:
                if page.tag.rsplit("}", 1)[-1] != "page":
                    continue
                try:
                    title = next(
                        child.text or "" for child in page if child.tag.rsplit("}", 1)[-1] == "title"
                    )
                    namespace = next(
                        child.text or "" for child in page if child.tag.rsplit("}", 1)[-1] == "ns"
                    )
                    page_id = next(
                        child.text or "" for child in page if child.tag.rsplit("}", 1)[-1] == "id"
                    )
                    revision = next(
                        child for child in page if child.tag.rsplit("}", 1)[-1] == "revision"
                    )
                    text_element = next(
                        child for child in revision if child.tag.rsplit("}", 1)[-1] == "text"
                    )
                    wiki_text = text_element.text or ""
                except StopIteration:
                    page.clear()
                    continue
                if (
                    namespace != "0"
                    or not page_id
                    or wiki_text.lstrip().casefold().startswith("#redirect")
                ):
                    page.clear()
                    continue
                cleaned = _wikitext_to_plain_text(wiki_text)
                if cleaned:
                    article_base_url = str(
                        (source.adapter_options or {}).get("article_base_url", source.homepage)
                    ).rstrip("/")
                    yield page_id, cleaned, source.category, {
                        "media_type": "text/plain",
                        "original_title": title,
                        "source_document_id": page_id,
                        "source_url": f"{article_base_url}/{urllib.parse.quote(title.replace(' ', '_'), safe=':_()')}",
                        "dump_format": "mediawiki-pages-articles-xml-bz2",
                        "license_scope": "source-level reviewed Wikimedia project text license",
                    }
                page.clear()

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
