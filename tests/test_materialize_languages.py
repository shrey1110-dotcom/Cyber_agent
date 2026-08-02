from __future__ import annotations

from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.materialize import materialize_archive
from cyber_agent.data_pipeline.materialize import _category, _language
from cyber_agent.data_pipeline.sources import SourceDefinition


def _source() -> SourceDefinition:
    return SourceDefinition.from_dict(
        {
            "source_name": "go-fixture",
            "exact_release_or_version": "v1",
            "download_location": "local://go-fixture",
            "publisher": "Fixture",
            "license": "Apache-2.0",
            "category": "cybersecurity",
            "retrieved_at": "2026-08-02T00:00:00+00:00",
            "local_research_source": True,
            "acquisition_enabled": False,
        }
    )


def test_materializer_recognizes_go_as_code_without_reformatting() -> None:
    source = _source()
    assert _language(Path("cmd/verify/main.go")) == "go"
    assert _category(source, Path("cmd/verify/main.go"), "\tif err != nil {\n\t\treturn err\n\t}") == "code"


def test_materializer_keeps_markdown_as_documentation_not_code() -> None:
    source = _source()
    path = Path("docs/secure-operation.md")
    assert _language(path) == "markdown"
    assert _category(source, path, "# Safe operation\n\nUse approved controls.") == "cybersecurity"


def test_materializer_rejects_empty_path_filter_output(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    source = SourceDefinition.from_dict(
        {
            "source_name": "empty-filter-fixture",
            "exact_release_or_version": "v1",
            "download_location": "https://approved.example/fixture.tar.gz",
            "publisher": "Fixture",
            "license": "Apache-2.0",
            "category": "general",
            "retrieved_at": "2026-08-02T00:00:00+00:00",
            "local_research_source": True,
            "acquisition_enabled": False,
            "data_location": "data/sources/empty-filter-fixture/manifest.jsonl",
            "adapter_options": {"extensions": [".md"], "path_prefixes": ["missing/"]},
        }
    )
    extracted = pipeline_project / "archive"
    extracted.mkdir()
    (extracted / "guide.md").write_text("# Guide\n\nUse approved controls.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="produced no eligible records"):
        materialize_archive(config, source, extracted)

    assert not (pipeline_project / "data" / "sources" / "empty-filter-fixture").exists()
