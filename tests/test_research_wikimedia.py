from __future__ import annotations

import bz2
from dataclasses import replace
from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import read_jsonl
from cyber_agent.data_pipeline.materialize import materialize_wikimedia_xml_bz2
from cyber_agent.data_pipeline.sources import SourceDefinition, SourceRegistry


def _source() -> SourceDefinition:
    return SourceDefinition.from_dict(
        {
            "source_name": "wiki-fixture",
            "exact_release_or_version": "20260701",
            "homepage": "https://example-wiki.invalid/",
            "download_location": "https://approved.example/wiki-20260701.xml.bz2",
            "publisher": "Example Wikimedia project",
            "license": "CC-BY-SA-4.0",
            "license_evidence_url": "https://example-wiki.invalid/terms",
            "allowed_use": "private local research only",
            "redistribution_status": "not cleared for redistribution",
            "attribution_requirements": "retain article URL and contributor-history link",
            "per_record_license_field": "source-level license with article URL",
            "review_status": "approved_for_pilot",
            "reviewed_by": "test reviewer",
            "reviewed_at": "2026-08-03T00:00:00+00:00",
            "content_categories": ["general"],
            "known_risks": ["fixture"],
            "category": "general",
            "adapter": "http_wikimedia_xml_bz2",
            "enabled": True,
            "retrieved_at": "2026-08-03T00:00:00+00:00",
            "local_research_source": True,
            "acquisition_enabled": True,
            "approved_domains": ["approved.example"],
            "collection": "research-150m",
            "data_location": "data/collections/research-150m/sources/wiki-fixture/manifest.jsonl",
            "adapter_options": {"article_base_url": "https://example-wiki.invalid/wiki"},
        }
    )


def _xml() -> bytes:
    return b"""<?xml version=\"1.0\"?>
<mediawiki xmlns=\"http://www.mediawiki.org/xml/export-0.11/\">
  <page><title>Network security</title><ns>0</ns><id>11</id><revision><id>12</id><text>== Network security ==
Network security protects systems. [[Firewall|Firewalls]] filter traffic.
{{Infobox|ignored=true}}
&lt;ref&gt;ignored&lt;/ref&gt;</text></revision></page>
  <page><title>Redirected</title><ns>0</ns><id>13</id><revision><id>14</id><text>#REDIRECT [[Network security]]</text></revision></page>
  <page><title>Talk page</title><ns>1</ns><id>15</id><revision><id>16</id><text>skip</text></revision></page>
</mediawiki>"""


def test_wikimedia_materializer_is_streamed_attributed_and_isolated(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project).for_collection("research-150m")
    archive = pipeline_project / "fixture.xml.bz2"
    archive.write_bytes(bz2.compress(_xml()))

    result = materialize_wikimedia_xml_bz2(config, _source(), archive, token_limit=10_000)

    assert result["documents"] == 1
    rows = read_jsonl(Path(result["manifest"]))
    assert rows[0]["license"] == "CC-BY-SA-4.0"
    assert rows[0]["source_url"] == "https://example-wiki.invalid/wiki/Network_security"
    assert rows[0]["metadata"]["source_document_id"] == "11"
    saved = Path(result["manifest"]).parent / rows[0]["path"]
    assert "Firewalls filter traffic." in saved.read_text(encoding="utf-8")
    assert "Infobox" not in saved.read_text(encoding="utf-8")
    assert not (pipeline_project / "data" / "sources" / "wiki-fixture").exists()


def test_wikimedia_materializer_rejects_dtd_and_decompression_overage(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project).for_collection("research-150m")
    dtd = pipeline_project / "dtd.xml.bz2"
    dtd.write_bytes(bz2.compress(b"<!DOCTYPE mediawiki [<!ENTITY x 'bad'>]><mediawiki/>"))
    with pytest.raises(ValueError, match="DTD or entity"):
        materialize_wikimedia_xml_bz2(config, _source(), dtd)

    small_budget = replace(config, pilot_budget=replace(config.pilot_budget, maximum_decompressed_bytes=80))
    archive = pipeline_project / "limited.xml.bz2"
    archive.write_bytes(bz2.compress(_xml()))
    with pytest.raises(ValueError, match="decompressed-byte limit"):
        materialize_wikimedia_xml_bz2(small_budget, _source(), archive)
    assert not (small_budget.paths.sources / "wiki-fixture").exists()


def test_collection_binding_prevents_accidental_default_pilot_ingestion(pipeline_project: Path) -> None:
    default_config = PipelineConfig.load(pipeline_project)
    default_registry = SourceRegistry.load(default_config.paths)
    with pytest.raises(ValueError, match="not active collection"):
        default_registry.require_downloadable("simplewiki-20260701-research", default_config.license_policy)

    research_config = default_config.for_collection("research-150m")
    research_registry = SourceRegistry.load(research_config.paths)
    source = research_registry.require_downloadable("simplewiki-20260701-research", research_config.license_policy)
    assert source.collection == "research-150m"
