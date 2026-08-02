from __future__ import annotations

from pathlib import Path

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
