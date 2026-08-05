from __future__ import annotations

from pathlib import Path

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.export import read_jsonl
from cyber_agent.data_pipeline.materialize import materialize_stackexchange_posts
from cyber_agent.data_pipeline.sources import SourceDefinition


def _source() -> SourceDefinition:
    return SourceDefinition.from_dict(
        {
            "source_name": "stack-fixture",
            "exact_release_or_version": "archive-fixture-v1",
            "homepage": "https://stackoverflow.com/",
            "download_location": "https://approved.example/stack-fixture.7z",
            "publisher": "Stack Exchange fixture",
            "data_location": "data/sources/stack-fixture/manifest.jsonl",
            "license": "MULTIPLE-SPDX-REQUIRED",
            "license_evidence_url": "https://stackoverflow.com/help/licensing",
            "allowed_use": "test-only local research",
            "redistribution_status": "not cleared",
            "attribution_requirements": "preserve canonical post provenance",
            "per_record_license_field": "CreationDate",
            "review_status": "approved_for_pilot",
            "reviewed_by": "test reviewer",
            "reviewed_at": "2026-08-04T00:00:00+00:00",
            "content_categories": ["code", "cybersecurity", "networking", "linux"],
            "known_risks": ["fixture"],
            "category": "code",
            "adapter": "http_stackexchange_posts_7z",
            "enabled": True,
            "retrieved_at": "2026-08-04T00:00:00+00:00",
            "local_research_source": True,
            "acquisition_enabled": True,
            "approved_domains": ["approved.example"],
            "adapter_options": {
                "site_base_url": "https://stackoverflow.com",
                "posts_member_name": "Posts.xml",
            },
        }
    )


def test_stackexchange_materializer_keeps_per_post_license_and_provenance(pipeline_project: Path) -> None:
    config = PipelineConfig.load(pipeline_project)
    xml = pipeline_project / "Posts.xml"
    xml.write_text(
        """<?xml version=\"1.0\" encoding=\"utf-8\"?>
<posts>
  <row Id=\"1\" PostTypeId=\"1\" CreationDate=\"2010-04-07T10:00:00.000\" Title=\"Read a file safely\" Tags=\"&lt;python&gt;&lt;pathlib&gt;\" Body=\"&lt;p&gt;Use pathlib to validate a workspace path.&lt;/p&gt;&lt;pre&gt;&lt;code&gt;path.resolve()&lt;/code&gt;&lt;/pre&gt;\" OwnerUserId=\"5\" />
  <row Id=\"2\" PostTypeId=\"2\" ParentId=\"1\" CreationDate=\"2019-05-02T10:00:00.000\" Body=\"&lt;p&gt;Reject parent traversal and null bytes before resolving paths.&lt;/p&gt;\" OwnerUserId=\"6\" />
</posts>""",
        encoding="utf-8",
    )

    result = materialize_stackexchange_posts(config, _source(), xml, token_limit=10_000)
    rows = read_jsonl(Path(result["manifest"]))

    assert [row["license"] for row in rows] == ["CC-BY-SA-2.5", "CC-BY-SA-4.0"]
    assert rows[0]["source_url"] == "https://stackoverflow.com/questions/1"
    assert rows[1]["source_url"] == "https://stackoverflow.com/a/2"
    assert rows[0]["metadata"]["code_provenance_kind"] == "stackexchange_post"
    assert rows[0]["metadata"]["post_creation_date"] == "2010-04-07T10:00:00.000"
    saved = Path(result["manifest"]).parent / rows[0]["path"]
    assert "path.resolve()" in saved.read_text(encoding="utf-8")
