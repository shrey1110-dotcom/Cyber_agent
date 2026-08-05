from __future__ import annotations

from pathlib import Path

import pytest

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.materialize import _category, _language, _selected_paths, materialize_archive
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


def test_materializer_permits_a_reviewed_build_path_override_only(pipeline_project: Path) -> None:
    source = SourceDefinition.from_dict(
        {
            "source_name": "build-path-fixture",
            "exact_release_or_version": "v1",
            "download_location": "local://build-path-fixture",
            "publisher": "Fixture",
            "license": "MIT",
            "category": "general",
            "retrieved_at": "2026-08-02T00:00:00+00:00",
            "local_research_source": True,
            "acquisition_enabled": False,
            "adapter_options": {
                "extensions": [".rst"],
                "path_prefixes": ["package/doc/build/"],
                "allow_path_parts": ["build"],
            },
        }
    )
    extracted = pipeline_project / "build-path-archive"
    document = extracted / "package" / "doc" / "build" / "guide.rst"
    document.parent.mkdir(parents=True)
    document.write_text("Guide\n=====\n\n" + "Reviewed documentation. " * 10, encoding="utf-8")
    assert list(_selected_paths(extracted, source)) == [document]

    invalid = SourceDefinition.from_dict(
        {
            "source_name": "bad-build-path-fixture",
            "exact_release_or_version": "v1",
            "download_location": "local://bad-build-path-fixture",
            "publisher": "Fixture",
            "license": "MIT",
            "category": "general",
            "retrieved_at": "2026-08-02T00:00:00+00:00",
            "local_research_source": True,
            "acquisition_enabled": False,
            "adapter_options": {"allow_path_parts": ["not-a-skip-rule"]},
        }
    )
    with pytest.raises(ValueError, match="not a normally skipped"):
        list(_selected_paths(extracted, invalid))
