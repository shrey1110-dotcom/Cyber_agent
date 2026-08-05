# Local research pilot corpus

This milestone prepares a private, auditable 1M–3M-token research corpus. It
does not clear the dataset or resulting weights for publication, and local use
does not remove copyright, license, attribution, privacy, or other obligations.
Downloaded data, materialized source files, snapshots, and candidates are
ignored by Git and must not be pushed.

## Mode and limits

`config/dataset_mode.json` fixes the mode to `local_research_only` with release,
weight-publication, and dataset-redistribution flags all false.
`config/pilot_budget.json` caps downloads at 1 GB, raw documents at 25,000,
clean documents at 15,000, estimated tokens at 3,000,000, each source at 5,000
documents and 1,000,000 tokens, archives at 50,000 files and 3 GB expanded,
requests at 30 seconds, and retries at three. Category targets are configurable
25/25/15/10/20/5 percentages for general, code, Linux, networking,
cybersecurity, and terminal data. Documents are never duplicated to meet a
target.

## Configured sources

`config/local_research_sources.json` contains only exact configured URLs or a
local generator identifier. Each record includes source name, publisher,
release/snapshot label, stated license or `REVIEW_REQUIRED`, primary category,
retrieval date, domains, provenance URL, and materializer. The initial set is:

- Python 3.13 documentation snapshot under the stated PSF terms;
- Requests v2.32.4 source under its stated Apache-2.0 license;
- HTTPX 0.28.1 source under its stated BSD-3-Clause license;
- Black 26.5.1 source under its stated MIT license;
- Moby `docker-v29.0.0` source under its stated Apache-2.0 license;
- Cosign v3.0.6 source under its stated Apache-2.0 license;
- FastAPI 0.115.12 English documentation under its stated MIT license;
- Trivy v0.66.0 documentation under its stated Apache-2.0 license;
- OpenTelemetry Collector v0.153.0 documentation under its stated Apache-2.0 license;
- YARA v4.5.5 documentation under its stated BSD-3-Clause license;
- Git 2.50.0 documentation and selected shell/configuration files, recorded as
  GPL-2.0-only and requiring review before publication;
- Linux man-pages 6.15, recorded as `REVIEW_REQUIRED` because file-level terms
  vary;
- MITRE ATT&CK STIX v17.1, recorded as `REVIEW_REQUIRED` with its terms URL;
- CWE 4.17 XML, recorded as `REVIEW_REQUIRED` with its terms URL;
- deterministic synthetic examples for the five closed-registry tools.

This configuration is not a legal conclusion. Records labeled
`REVIEW_REQUIRED`, and every included source while `release_cleared` is false,
must be reviewed or removed before open-weight publication.

Only enabled records with review status `approved_for_pilot` or
`approved_for_production` can be acquired or ingested. A previously
materialized manifest does not override a later disabled or pending status.

## Acquisition

No arbitrary URL is accepted by the CLI. One confirmation gates the entire
configured network run; separate per-source human approval is not required.

```bash
python -m cyber_agent.data_pipeline.cli validate-sources
python -m cyber_agent.data_pipeline.cli acquire-pilot \
  --mode local_research_only \
  --target-tokens 3000000 \
  --seed 42 \
  --confirm-download
```

Without `--confirm-download`, the command fails before writing acquisition
outputs. The downloader retains the Phase 3.5 HTTPS/domain/redirect,
content-length, budget, timeout, retry, resume, rate-limit, checksum, temporary
file, and atomic rename protections. ZIP/TAR expansion rejects traversal and
all excess file-count, expanded-byte, and compression-ratio limits; it skips
and counts link and special-file entries rather than creating or following them.
Downloaded code and scripts are never executed.

Materializers accept only bounded UTF-8 text. They reject minified, generated,
vendored, hidden dependency, extremely large, binary, and malformed files.
Code records preserve repository, revision, original path, and the stated
per-record terms label. STIX and CWE inputs are split into reviewable records
rather than treated as one oversized document.

The safe synthetic-only path does not require network confirmation:

```bash
python -m cyber_agent.data_pipeline.cli acquire-pilot \
  --mode local_research_only --target-tokens 3000000 --seed 42 \
  --source synthetic-safe-tool-examples-v3
```

## Processing and inspection

```bash
python -m cyber_agent.data_pipeline.cli run-all --seed 42 --force
python -m cyber_agent.data_pipeline.cli report
python -m cyber_agent.data_pipeline.cli show-samples --seed 42 --per-category 2
python -m cyber_agent.data_pipeline.cli freeze-snapshot \
  --name cyber-local-pilot-v1 --seed 42
```

`run-all` performs ingestion, extraction/normalization, unchanged secret and
personal-data filtering, quality scoring, exact/near deduplication,
deterministic balancing, group-aware splitting, exports, and reports. Rejection
reports contain codes and hashed provenance, never complete rejected secrets.

The immutable snapshot includes `LOCAL_RESEARCH_ONLY.txt`, checksums, source
and terms manifests, reports, and train/validation/test manifests. Its manifest
sets `fixture_artifact=false`, `local_research_only=true`,
`release_cleared=false`, `production_ready=false`, and
`weight_publication_allowed=false` for a real pilot.

## Tokenizer candidates

```bash
python -m cyber_agent.tokenizer.cli train-pilot-candidates \
  --snapshot cyber-local-pilot-v1
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-local-pilot-v1 --candidate 16000
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-local-pilot-v1 --candidate 24000
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-local-pilot-v1 --candidate 32000
python -m cyber_agent.tokenizer.cli compare \
  --snapshot cyber-local-pilot-v1
```

Training reads only the frozen training manifest. Validation, test, and
representative technical fixtures are evaluation-only. A sufficiently large
local corpus may receive a local research recommendation, but that
recommendation never clears publication and never triggers final export.
