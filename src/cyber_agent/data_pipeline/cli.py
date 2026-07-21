"""Command-line interface for each resumable Phase 2 pipeline stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.deduplicate import run_deduplicate
from cyber_agent.data_pipeline.export import run_export
from cyber_agent.data_pipeline.ingest import run_ingest
from cyber_agent.data_pipeline.normalize import run_clean
from cyber_agent.data_pipeline.reports import run_report
from cyber_agent.data_pipeline.sources import SourceRegistry
from cyber_agent.data_pipeline.split import run_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable Cyber Agent training-data pipeline")
    parser.add_argument("--project-root", type=Path, help="project root containing config/ and data/")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-sources", help="validate source allowlist and license policy")
    ingest = subparsers.add_parser("ingest", help="ingest one or more enabled local sources")
    ingest.add_argument("--source", action="append", dest="sources", help="allowlisted source name")
    ingest.add_argument("--force", action="store_true")
    clean = subparsers.add_parser("clean", help="extract, normalize, filter, and score documents")
    clean.add_argument("--force", action="store_true")
    deduplicate = subparsers.add_parser("deduplicate", help="remove exact and near duplicates")
    deduplicate.add_argument("--force", action="store_true")
    split = subparsers.add_parser("split", help="create deterministic dataset splits")
    split.add_argument("--seed", type=int, default=None)
    split.add_argument("--force", action="store_true")
    export = subparsers.add_parser("export", help="write final JSONL and audit manifests")
    export.add_argument("--force", action="store_true")
    subparsers.add_parser("report", help="generate aggregate dataset and license reports")
    run_all = subparsers.add_parser("run-all", help="run all stages in dependency order")
    run_all.add_argument("--seed", type=int, default=None)
    run_all.add_argument("--source", action="append", dest="sources")
    run_all.add_argument("--force", action="store_true")
    return parser


def execute(args: argparse.Namespace) -> Any:
    config = PipelineConfig.load(args.project_root)
    if args.command == "validate-sources":
        registry = SourceRegistry.load(config.paths)
        errors = registry.validate(config.license_policy)
        if errors:
            raise ValueError("; ".join(errors))
        return {
            "status": "valid",
            "configured_sources": len(registry.all_sources()),
            "enabled_sources": [source.source_name for source in registry.enabled_sources(config.license_policy)],
        }
    if args.command == "ingest":
        return run_ingest(config, args.sources, force=args.force)
    if args.command == "clean":
        return run_clean(config, force=args.force)
    if args.command == "deduplicate":
        return run_deduplicate(config, force=args.force)
    if args.command == "split":
        return run_split(config, seed=args.seed, force=args.force)
    if args.command == "export":
        return run_export(config, force=args.force)
    if args.command == "report":
        return run_report(config)
    if args.command == "run-all":
        results = [
            run_ingest(config, args.sources, force=args.force),
            run_clean(config, force=args.force),
            run_deduplicate(config, force=args.force),
            run_split(config, seed=args.seed, force=args.force),
            run_export(config, force=args.force),
        ]
        results.append({"stage": "report", "status": "complete", "summary": run_report(config)})
        return {"status": "complete", "stages": results}
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except KeyboardInterrupt:
        print(json.dumps({"status": "failure", "error": "interrupted"}), file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"status": "failure", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

