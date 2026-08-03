"""Command-line interface for each resumable Phase 2 pipeline stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.config import PipelineConfig
from cyber_agent.data_pipeline.acquisition import acquire_reviewed_source
from cyber_agent.data_pipeline.balance import run_balance
from cyber_agent.data_pipeline.deduplicate import run_deduplicate
from cyber_agent.data_pipeline.export import run_export
from cyber_agent.data_pipeline.ingest import run_ingest
from cyber_agent.data_pipeline.normalize import run_clean
from cyber_agent.data_pipeline.reports import representative_samples, run_report
from cyber_agent.data_pipeline.pilot_acquisition import acquire_pilot
from cyber_agent.data_pipeline.sources import SourceRegistry
from cyber_agent.data_pipeline.snapshot import freeze_snapshot, snapshot_directory_name, verify_snapshot
from cyber_agent.provenance import generate_provenance_attestation
from cyber_agent.data_pipeline.split import run_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable Cyber Agent training-data pipeline")
    parser.add_argument("--project-root", type=Path, help="project root containing config/ and data/")
    parser.add_argument(
        "--collection",
        help=(
            "isolated named collection; uses the separately configured research "
            "budget and never overwrites the default pilot data"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-sources", help="validate source allowlist and license policy")
    acquire = subparsers.add_parser("acquire", help="download one explicitly selected, reviewed source")
    acquire.add_argument("--source", required=True, help="exact allowlisted source name")
    acquire.add_argument("--confirm-download", action="store_true")
    acquire_pilot_parser = subparsers.add_parser("acquire-pilot", help="acquire configured local-research pilot sources")
    acquire_pilot_parser.add_argument("--mode", required=True, choices=("local_research_only",))
    acquire_pilot_parser.add_argument("--target-tokens", required=True, type=int)
    acquire_pilot_parser.add_argument("--seed", type=int, default=42)
    acquire_pilot_parser.add_argument("--source", action="append", dest="sources")
    acquire_pilot_parser.add_argument("--confirm-download", action="store_true")
    acquire_research_parser = subparsers.add_parser(
        "acquire-research",
        help="acquire only explicitly reviewed sources into an isolated named research collection",
    )
    acquire_research_parser.add_argument("--target-tokens", required=True, type=int)
    acquire_research_parser.add_argument("--seed", type=int, default=42)
    acquire_research_parser.add_argument("--source", action="append", dest="sources")
    acquire_research_parser.add_argument("--confirm-download", action="store_true")
    ingest = subparsers.add_parser("ingest", help="ingest one or more enabled local sources")
    ingest.add_argument("--source", action="append", dest="sources", help="allowlisted source name")
    ingest.add_argument("--force", action="store_true")
    clean = subparsers.add_parser("clean", help="extract, normalize, filter, and score documents")
    clean.add_argument("--force", action="store_true")
    deduplicate = subparsers.add_parser("deduplicate", help="remove exact and near duplicates")
    deduplicate.add_argument("--force", action="store_true")
    balance = subparsers.add_parser("balance", help="apply deterministic source and category caps")
    balance.add_argument("--seed", type=int, default=None)
    balance.add_argument("--force", action="store_true")
    split = subparsers.add_parser("split", help="create deterministic dataset splits")
    split.add_argument("--seed", type=int, default=None)
    split.add_argument("--force", action="store_true")
    export = subparsers.add_parser("export", help="write final JSONL and audit manifests")
    export.add_argument("--force", action="store_true")
    subparsers.add_parser("report", help="generate aggregate dataset and license reports")
    samples = subparsers.add_parser("show-samples", help="show deterministic accepted samples by category")
    samples.add_argument("--seed", type=int, default=42)
    samples.add_argument("--per-category", type=int, default=2)
    run_all = subparsers.add_parser("run-all", help="run all stages in dependency order")
    run_all.add_argument("--seed", type=int, default=None)
    run_all.add_argument("--source", action="append", dest="sources")
    run_all.add_argument("--force", action="store_true")
    freeze = subparsers.add_parser("freeze-snapshot", help="freeze an immutable balanced pilot snapshot")
    freeze.add_argument("--name", required=True)
    freeze.add_argument("--version", type=int, default=1)
    freeze.add_argument("--seed", type=int, default=None)
    verify = subparsers.add_parser("verify-snapshot", help="verify immutable snapshot checksums")
    verify.add_argument("--name", required=True)
    verify.add_argument("--version", type=int, default=1)
    attest = subparsers.add_parser("attest-snapshot", help="create a local provenance attestation without uploading")
    attest.add_argument("--name", required=True)
    attest.add_argument("--version", type=int, default=1)
    attest.add_argument("--signer-identity")
    attest.add_argument("--detached-signature", type=Path)
    return parser


def execute(args: argparse.Namespace) -> Any:
    config = PipelineConfig.load(args.project_root).for_collection(args.collection)
    if args.command == "validate-sources":
        registry = SourceRegistry.load(config.paths)
        errors = registry.validate(config.license_policy)
        if errors:
            raise ValueError("; ".join(errors))
        return {
            "status": "valid",
            "configured_sources": len(registry.all_sources()),
            "enabled_sources": [source.source_name for source in registry.enabled_sources(config.license_policy)],
            "pending_sources": [
                source.source_name for source in registry.all_sources() if source.review_status == "pending"
            ],
            "dataset_mode": config.dataset_mode.to_dict(),
            "review_required_sources": [
                source.source_name for source in registry.all_sources()
                if source.license == "REVIEW_REQUIRED"
            ],
        }
    if args.command == "acquire":
        if not args.confirm_download:
            raise ValueError("network acquisition requires explicit --confirm-download")
        return acquire_reviewed_source(config, args.source)
    if args.command == "acquire-pilot":
        return acquire_pilot(
            config,
            mode=args.mode,
            target_tokens=args.target_tokens,
            seed=args.seed,
            confirm_download=args.confirm_download,
            source_names=args.sources,
        )
    if args.command == "acquire-research":
        if not args.collection:
            raise ValueError("acquire-research requires --collection to isolate its outputs")
        return acquire_pilot(
            config,
            mode="local_research_only",
            target_tokens=args.target_tokens,
            seed=args.seed,
            confirm_download=args.confirm_download,
            source_names=args.sources,
        )
    if args.command == "ingest":
        return run_ingest(config, args.sources, force=args.force)
    if args.command == "clean":
        return run_clean(config, force=args.force)
    if args.command == "deduplicate":
        return run_deduplicate(config, force=args.force)
    if args.command == "balance":
        return run_balance(config, seed=args.seed, force=args.force)
    if args.command == "split":
        return run_split(config, seed=args.seed, force=args.force)
    if args.command == "export":
        return run_export(config, force=args.force)
    if args.command == "report":
        return run_report(config)
    if args.command == "show-samples":
        if args.per_category < 1 or args.per_category > 10:
            raise ValueError("per-category sample count must be between 1 and 10")
        return {"seed": args.seed, "samples": representative_samples(config, seed=args.seed, per_category=args.per_category)}
    if args.command == "run-all":
        results = [
            run_ingest(config, args.sources, force=args.force),
            run_clean(config, force=args.force),
            run_deduplicate(config, force=args.force),
            run_balance(config, seed=args.seed, force=args.force),
            run_split(config, seed=args.seed, force=args.force),
            run_export(config, force=args.force),
        ]
        results.append({"stage": "report", "status": "complete", "summary": run_report(config)})
        return {"status": "complete", "stages": results}
    if args.command == "freeze-snapshot":
        return freeze_snapshot(config, name=args.name, seed=args.seed, version=args.version)
    if args.command == "verify-snapshot":
        path = config.paths.snapshots / snapshot_directory_name(args.name, args.version)
        manifest = verify_snapshot(path)
        return {"status": "valid", "snapshot": args.name, "snapshot_content_hash": manifest["snapshot_content_hash"]}
    if args.command == "attest-snapshot":
        directory_name = snapshot_directory_name(args.name, args.version)
        path = config.paths.snapshots / directory_name
        output = generate_provenance_attestation(
            path,
            artifact_type="frozen_dataset_snapshot",
            signer_identity=args.signer_identity,
            detached_signature=args.detached_signature,
            output_path=config.paths.snapshots / f"{directory_name}.attestation.json",
        )
        return {"status": "complete", "attestation": str(output), "uploaded": False}
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
