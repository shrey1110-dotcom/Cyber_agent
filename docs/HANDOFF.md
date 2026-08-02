# cyber-agent engineering handoff

This document is the continuation guide for an LLM or engineer taking over the
`cyber-agent` repository. It describes the implementation that exists now, the
generated local state that was observed, the invariants that must not be
weakened, and the work that remains before a production model or public dataset
release.

Read this document before editing code. Treat generated data and snapshots as
user-owned state. Do not delete, reset, overwrite, publish, or download data as
part of routine maintenance.

## Current status in one paragraph

Phases 1, 2, 3, and the Phase 3.5 tooling milestone are implemented. Phase 1
is a local agent shell with strict JSON actions, exactly five tools, filesystem
policy checks, Docker-only execution, and audit logging. Phase 2 is a
license-aware cleaning, filtering, deduplication, splitting, export, and report
pipeline. Phase 3 trains deterministic byte-level BPE tokenizer candidates and
provides a stable loader. Phase 3.5 adds source-review records, bounded local
research acquisition, safe archive extraction, deterministic balancing,
immutable snapshots, snapshot-specific tokenizer candidates, model-budget
estimates, candidate comparison, trusted prompt serialization, and checksums.

The current working data directories contain a still-tiny synthetic fixture
pipeline: 43 raw records, 42 extracted records, 41 cleaned records, and 39
deduplicated/balanced records (approximately 62,468 provisional tokens). The
currently frozen snapshots are older and contain only four retained documents
and approximately 404 provisional tokens. The snapshots are `pilot_only`; they
are not production datasets or production tokenizers. No transformer, MLX
training loop, hosted LLM integration, or public data release exists.

## Verified state at handoff

The working branch was `main`, with `origin/main` at commit `a90bb87` before
this handoff document is added. The complete test suite passed:

```text
97 passed in 1.04s
```

The prior 86 Phase 1–3 tests are included in that run and still pass. The
working tree contains generated, ignored/local data directories from fixture
demonstrations:

```text
data/downloads/
data/generated/
data/sources/
artifacts/datasets/
```

Do not assume those generated files belong in Git. `git status --short` reports
these directories because generated snapshots and source material are local
artifacts. The intended policy is to keep large/generated data out of Git.

The current extracted/materialized local data is specifically:

| Stage/path | Records | Meaning |
| --- | ---: | --- |
| `data/raw/documents.jsonl` | 43 | Ingested raw records, including records later rejected. |
| `data/extracted/documents.jsonl` | 42 | UTF-8/text extraction output; one malformed/raw record was rejected before or during extraction. |
| `data/cleaned/documents.jsonl` | 41 | Normalized, quality-checked records before deduplication. |
| `data/cleaned/deduplicated.jsonl` | 39 | Two exact/near duplicates removed. |
| `data/cleaned/balanced.jsonl` | 39 | Deterministic balancing output; no records excluded in this tiny run. |
| `data/splits/train.jsonl` | 39 | Current tokenizer-training input. Validation and test are empty for this fixture. |

The extracted source files are local synthetic JSON documents under
`data/sources/synthetic-safe-tool-examples-v3/documents/`. The acquisition
manifest records 35 generated documents, 69,815 provisional source tokens, and
zero downloaded bytes. `data/downloads/` is empty. No external Python, Git,
Linux, MITRE, CWE, NIST, FineWeb, or other remote corpus has been downloaded.

The observed frozen snapshots are:

| Snapshot | Documents | Estimated tokens | Train/validation/test | Status |
| --- | ---: | ---: | --- | --- |
| `cyber-pilot-v1` | 4 accepted, 4 rejected | 404 | 4 / 0 / 0 | `pilot_only` |
| `cyber-pilot-v1.v2` | 4 accepted, 4 rejected | 404 | 4 / 0 / 0 | `pilot_only` |
| `cyber-pilot-v3` | 39 accepted, 4 rejected | 62,468 | 39 / 0 / 0 | `pilot_only` |

Both snapshots are local-research-only, release-cleared is false, dataset
redistribution is false, and model-weight publication is false. They are
immutable: the same name/version cannot be frozen twice, and checksum changes
make verification fail. Their embedded provenance records refer to the Git
state that existed when each snapshot was created; a future snapshot must be
created after any input/configuration change.

The snapshot-specific tokenizer candidates are under
`artifacts/tokenizers/candidates/cyber-pilot-v1/16000`, `24000`, and `32000`.
All three requested sizes produce the same actual fixture vocabulary of 458
tokens. They are marked as fixture artifacts, use the same frozen training
manifest, decode exactly on the fixture evaluation set, and have zero unknown
tokens there. Candidate comparison reports `insufficient_evidence` and makes
no recommendation because the corpus and held-out evidence are too small.
The `cyber-pilot-v1` candidates were trained from the older four-document
snapshot. The `cyber-pilot-v3` candidates now use the newer 39-record frozen
training manifest and have actual vocabulary size 1,003 for each requested
16K/24K/32K candidate. They still remain fixture artifacts: the comparison
report has 16 evaluation examples, 63,918 exact training tokens, zero unknowns,
100% round-trip accuracy, but no recommendation because the configured minimum
evidence is 1,000 evaluation documents and 10M estimated training tokens.

## System architecture

The intended end-to-end system is:

```text
user request
  -> ModelBackend.generate(messages)
  -> strict action JSON parser
  -> closed tool registry and path/argument policy
  -> fresh hardened Docker container
  -> structured tool result
  -> model receives the result
  -> final answer or another bounded tool call
```

The data and tokenizer path is separate from runtime execution:

```text
reviewed source records
  -> optional explicitly confirmed acquisition
  -> raw/local source manifests
  -> ingest
  -> extraction and normalization
  -> sensitive-data and quality rejection
  -> exact/near deduplication
  -> deterministic balancing
  -> duplicate-group-aware train/validation/test split
  -> immutable frozen snapshot
  -> tokenizer candidates trained on frozen train only
  -> held-out evaluation and comparison
  -> explicit pilot/final export
  -> future MLX model training
```

The model is intentionally not implemented. The only current model boundary is
the `ModelBackend` protocol in `src/cyber_agent/model.py`. A future local MLX
backend must implement that protocol and must not bypass the parser, policy,
runtime, or audit layers.

## Phase status

### Phase 1 — safe agent shell: complete

Implemented:

- Strict JSON model actions: `tool_call` or `final_answer`.
- Tool results with `success`, `rejected`, or `failure` status.
- Exactly five registered tools: `list_files`, `read_file`,
  `check_processes`, `check_ports`, and `run_tests`.
- No arbitrary shell-command tool and no `shell=True`.
- Workspace path validation for null bytes, traversal, foreign absolute paths,
  backslashes, and symlink escapes.
- Fresh non-root Docker execution with `--network none`, read-only root and
  workspace mount, dropped capabilities, `no-new-privileges`, PID/memory/CPU
  limits, a constrained `/tmp`, and wall-clock timeouts.
- Structured audit events for requests, policy rejection, execution start,
  runtime failures, cleanup, and completion.
- A deterministic mock backend for tests only. It is not a trained model and
  must never be described as the production model.

### Phase 2 — auditable training-data pipeline: complete for fixtures

Implemented:

- Strongly typed document schemas and source/license policy checks.
- Local-manifest ingestion with source allowlist and exact release checks.
- UTF-8 validation, HTML extraction, Unicode normalization, code-preserving
  cleaning, quality scoring, repetition/markup/navigation checks, and bounded
  record sizes.
- Secret and personal-data detection. Rejection records retain safe reasons and
  provenance, not complete rejected secrets.
- Exact SHA-256 deduplication and near-duplicate grouping.
- Deterministic 98/1/1 splitting with duplicate groups kept together and hash
  leakage checks.
- Atomic JSON/JSONL outputs and stage markers for resumability.
- Dataset, license, rejection, duplicate, source, and split reports.

The fixture pipeline is not a production corpus. Empty validation/test splits
are expected for the tiny fixture and are not evidence of adequate evaluation.

### Phase 3 — tokenizer pipeline: complete for fixtures

Implemented:

- Deterministic byte-level BPE training with the local `tokenizers` library.
- Stable special-token ordering and IDs.
- Train-split-only corpus validation, including manifest and license checks.
- Candidate artifacts, resolved configuration, artifact hashes, vocabulary and
  merge exports, and a model-independent `CyberTokenizer` loader.
- Evaluation across prose, code, shell, JSON/YAML, logs, paths, network
  addresses, CVE/CWE/MITRE identifiers, and Unicode.
- Candidate comparison that does not automatically select or export a final
  tokenizer.

The original Phase 3 fixture tokenizer is at
`artifacts/tokenizers/candidates/512`; it is also non-production. Phase 3.5
adds the snapshot-specific 16K/24K/32K request workflow described below.

### Phase 3.5 — pilot acquisition and candidate evaluation: tooling complete,
production corpus not complete

Implemented:

- Exact source metadata and review-status enforcement for the main allowlist.
- Separate `local_research_only` mode with explicit non-publication flags.
- Bounded download helpers with domain restrictions, redirect checks,
  credentials rejection, user-agent, timeout, retries, rate limiting,
  content-length and byte budgets, resume files, optional SHA-256, and atomic
  publication.
- ZIP/TAR extraction protections against traversal, links, special files,
  excessive file counts, decompression size, and compression bombs.
- Local materializers for text archives, STIX JSON, CWE XML, and deterministic
  synthetic safe tool examples. Downloaded code is never executed.
- Provisional token estimation using `max(1, ceil(UTF-8 bytes / 4))`, explicitly
  labeled as an estimate.
- Deterministic source/category balancing without document duplication, with
  duplicate groups kept together and exclusion reasons recorded.
- Immutable snapshot creation with required manifests, reports, checksums,
  configuration hashes, Git commit, readiness flags, and optional local
  provenance attestation.
- Snapshot-specific 16K/24K/32K candidate training and evaluation, exact
  candidate counts beside provisional counts, sequence percentiles,
  programming-language metrics, fragmentation inspection, source/category
  coverage, and evidence-gated comparison.
- A configurable 50M-parameter vocabulary budget estimate for tied and untied
  embeddings.
- Canonical trusted prompt serialization for system, user, assistant, tool-call,
  tool-result, terminal, retrieved-document, and code components.
- Explicit confirmation required for final export; default status is
  `pilot_only`.

Not complete:

- No real 10M–30M-token pilot has been acquired or legally cleared.
- No external source has been approved for production release.
- No production tokenizer has been selected or exported.
- No transformer architecture or MLX training exists.

## File-by-file map

The following map is intentionally explicit so a new LLM can locate behavior
without guessing.

### Runtime and agent shell

| File | Responsibility |
| --- | --- |
| `src/cyber_agent/__init__.py` | Package marker and public package metadata. |
| `src/cyber_agent/schema.py` | Wire-format dataclasses, strict model-action parsing, tool-result serialization, and message roles. |
| `src/cyber_agent/model.py` | Abstract `ModelBackend` protocol and temporary deterministic mock backend. |
| `src/cyber_agent/agent.py` | Bounded model/tool loop, model-output validation, message accumulation, and agent audit events. |
| `src/cyber_agent/policy.py` | Workspace path resolution and traversal/symlink/absolute-path policy. |
| `src/cyber_agent/tools.py` | Closed five-tool registry, exact argument validators, timeout declarations, and tool result handling. |
| `src/cyber_agent/runtime.py` | Docker-only subprocess invocation and hardened container arguments. |
| `src/cyber_agent/sandbox_worker.py` | In-container fixed implementations for the five tools; it accepts structured requests, not arbitrary commands. |
| `src/cyber_agent/audit.py` | Structured JSON-line audit event helper. |
| `src/cyber_agent/cli.py` | User-facing Phase 1 command-line entry point and demos. |
| `Dockerfile` | Non-root sandbox image definition used by `DockerRuntime`. |

### Phase 2 and Phase 3.5 data pipeline

| File | Responsibility |
| --- | --- |
| `src/cyber_agent/data_pipeline/__init__.py` | Data-pipeline package exports. |
| `src/cyber_agent/data_pipeline/schemas.py` | Typed documents, split assignments, categories, hashes, timestamps, and canonical record validation. |
| `src/cyber_agent/data_pipeline/config.py` | Pipeline paths, license policy, dataset mode, pilot budgets, category targets, and configuration fingerprints. |
| `src/cyber_agent/data_pipeline/sources.py` | Source definitions, review statuses, source registry, allowlist, exact-release checks, and local-research source loading. |
| `src/cyber_agent/data_pipeline/acquisition.py` | Explicit reviewed downloads, resumable transfers, budgets, redirects, checksums, rate limits, and safe ZIP/TAR extraction. |
| `src/cyber_agent/data_pipeline/pilot_acquisition.py` | Orchestrates synthetic and remote local-research acquisition under the configured pilot limits. |
| `src/cyber_agent/data_pipeline/synthetic.py` | Deterministic safe examples for only the five closed agent tools. |
| `src/cyber_agent/data_pipeline/materialize.py` | Converts reviewed archives/STIX/CWE inputs into bounded per-record local manifests while preserving provenance and license fields. |
| `src/cyber_agent/data_pipeline/ingest.py` | Reads source manifests, verifies source/release/license metadata, validates content, and writes raw records/rejections. |
| `src/cyber_agent/data_pipeline/extract.py` | Text and HTML extraction primitives. |
| `src/cyber_agent/data_pipeline/normalize.py` | Unicode/whitespace/control-character normalization, code preservation, sensitive-data calls, quality rejection, and cleaned records. |
| `src/cyber_agent/data_pipeline/quality.py` | Deterministic quality score and hard quality checks. |
| `src/cyber_agent/data_pipeline/sensitive_data.py` | Secret, credential, email, phone, and personal-data pattern detection. |
| `src/cyber_agent/data_pipeline/deduplicate.py` | Exact content hashes, near-duplicate grouping, and duplicate reports. |
| `src/cyber_agent/data_pipeline/balance.py` | Provisional token estimation and deterministic source/category caps and target balancing. |
| `src/cyber_agent/data_pipeline/split.py` | Seeded duplicate-group-aware train/validation/test assignment and leakage validation. |
| `src/cyber_agent/data_pipeline/export.py` | Atomic JSON/JSONL writes, stage fingerprints/markers, and Phase 2 export manifests. |
| `src/cyber_agent/data_pipeline/reports.py` | Dataset, license, rejection, duplicate, source concentration, and representative-sample reports. |
| `src/cyber_agent/data_pipeline/snapshot.py` | Immutable balanced snapshot creation, required snapshot files, checksums, readiness flags, and verification. |
| `src/cyber_agent/data_pipeline/cli.py` | Data validation, acquisition, processing, balancing, splitting, reporting, snapshot, verification, and attestation commands. |
| `src/cyber_agent/provenance.py` | Local provenance/attestation metadata and optional detached-signature recording; never uploads anything. |

### Tokenizer

| File | Responsibility |
| --- | --- |
| `src/cyber_agent/tokenizer/__init__.py` | Public tokenizer exports. |
| `src/cyber_agent/tokenizer/config.py` | Typed tokenizer configuration, special-token contract, candidate sizes, paths, and configuration hash. |
| `src/cyber_agent/tokenizer/corpus.py` | Validates the legacy Phase 2 train split and provides streamed corpus plans; also loads frozen snapshot train manifests with train/held-out isolation checks. |
| `src/cyber_agent/tokenizer/trainer.py` | Trains one legacy byte-level BPE candidate from the Phase 2 train split. |
| `src/cyber_agent/tokenizer/pilot.py` | Trains/evaluates/compares snapshot candidates, computes fragmentation and exact counts, gates recommendations, and performs confirmed snapshot export. |
| `src/cyber_agent/tokenizer/artifacts.py` | Writes candidate metadata, special-token maps, vocabulary/merge files, artifact hashes, and transactional legacy/final exports. |
| `src/cyber_agent/tokenizer/loader.py` | Stable Python encoding/decoding API, batch padding, masks, BOS/EOS insertion, truncation, and literal-special-token safety. |
| `src/cyber_agent/tokenizer/evaluator.py` | Legacy candidate evaluation and comparison reports. |
| `src/cyber_agent/tokenizer/model_budget.py` | Architectural estimates for embedding/output vocabulary costs under tied and untied weights. |
| `src/cyber_agent/tokenizer/serialization.py` | Canonical `cyber-agent-trusted-prompt-v1` serializer; the only component allowed to insert trusted control-token IDs. |
| `src/cyber_agent/tokenizer/cli.py` | Legacy and snapshot tokenizer commands, model-budget command, inspection, serialization, export, and local attestation. |

### Configuration, documentation, fixtures, and tests

| File or directory | Responsibility |
| --- | --- |
| `config/approved_sources.json` | Main source allowlist. `sample` is the only `approved_for_pilot` and enabled fixture; external placeholders are pending and disabled. |
| `config/local_research_sources.json` | Exact local-research source/release records and adapter settings. This is not production approval. |
| `config/license_policy.json` | License identifiers and allowed/review-required/denied policy statuses. |
| `config/dataset_mode.json` | `local_research_only` mode and publication/redistribution false flags. |
| `config/pilot_budget.json` | Download, document, estimated-token, source, category, archive, timeout, retry, and target limits. |
| `config/data_pipeline.json` | Phase 2 cleaning, quality, deduplication, split, and seed configuration. |
| `config/tokenizer.json` | Byte-level BPE settings, 16K/24K/32K candidates, 24K default, special tokens, and train path. |
| `fixtures/sample_corpus/` | Tiny accepted/rejected Phase 2 fixture corpus, manifest, and CC0 fixture license. |
| `fixtures/tokenizer_evaluation.jsonl` | Held-out tokenizer examples for commands, code, JSON/YAML, paths, addresses, identifiers, logs, and Unicode. |
| `docs/data_pipeline.md` | Phase 2 source, cleaning, licensing, deduplication, split, and report guide. |
| `docs/tokenizer.md` | Phase 3 byte-level tokenizer design, commands, special-token contract, and MLX handoff. |
| `docs/local_research_pilot.md` | Phase 3.5 acquisition, local-research restrictions, candidate workflow, and publication warnings. |
| `docs/pilot_corpus.md` | Phase 3.5 review, acquisition controls, balancing, snapshot, candidate, serialization, and export guide. |
| `docs/HANDOFF.md` | This LLM-oriented continuation report. |
| `README.md` | Human entry point with project status, setup, security controls, and common commands. |
| `SECURITY.md` | Minimum security floor and future permission/authorization boundaries. |
| `.gitignore` | Keeps generated data, snapshots, and tokenizer artifacts out of Git while retaining directory sentinels. |
| `tests/test_agent.py`, `test_policy.py`, `test_runtime.py`, `test_tools.py`, `test_worker.py`, `test_schema.py` | Phase 1 agent, policy, Docker invocation, tool, worker, and wire-schema coverage. |
| `tests/test_data_cleaning.py`, `test_data_deduplicate_split.py`, `test_data_io_reports.py`, `test_data_pipeline_integration.py`, `test_data_schemas_config.py` | Phase 2 data pipeline coverage. |
| `tests/test_tokenizer_config_corpus.py`, `test_tokenizer_training_loader.py`, `test_tokenizer_evaluation_cli.py` | Phase 3 tokenizer corpus, training, loader, evaluation, CLI, and determinism coverage. |
| `tests/test_local_research_pilot.py` | Local-research mode, synthetic generation, source visibility, and acquisition confirmation coverage. |
| `tests/test_phase35_sources_acquisition.py` | Review/release enforcement, download budget/resume/redirect/atomicity, and archive safety coverage. |
| `tests/test_phase35_balance_snapshot.py` | Deterministic balancing, provisional estimator, immutable snapshots, checksum verification, and attestation coverage. |
| `tests/test_phase35_tokenizer_serialization.py` | Frozen candidates, exact metrics, evidence-gated comparison, model budget, trusted serialization, injection safety, and export confirmation coverage. |
| `tests/conftest.py` | Builds isolated temporary Phase 2/tokenizer projects for integration tests. |
| `pyproject.toml` | Package metadata, Python version, `tokenizers` dependency, pytest settings, and CLI entry points. |
| `uv.lock` | Reproducible dependency lockfile. |

## Source and licensing state

The main allowlist deliberately separates approval from public visibility.

### Approved for the fixture pilot

`sample` is pinned to `fixture-pilot-v1`, has a local CC0-1.0 fixture license,
is reviewed by `fixture-maintainer`, and is enabled. It is synthetic/tiny and is
not representative of the intended model corpus.

### Pending or not production-approved

The following remain pending, disabled placeholders in the main allowlist:

- FineWeb-Edu: exact release and rights for underlying web documents.
- NIST publications: publication-level review and third-party figures/standards.
- MITRE ATT&CK: exact release terms and attribution.
- CWE: exact release terms and attribution.
- Linux documentation: file-level SPDX and authorship review.
- Bash/GNU documentation: exact version and copyleft/GFDL obligations.
- Python documentation: pinned release and third-party/example notices.
- Git documentation: exact release and GPL/documentation review.
- Public/permissively licensed code: per-repository, revision, file, SPDX, and
  notice verification.

The local-research configuration contains exact records for Python docs 3.13,
Requests 2.32.4, Git 2.50.0, Linux man-pages 6.15, MITRE ATT&CK STIX v17.1,
CWE 4.17, and synthetic examples. Some carry ordinary licenses and some carry
`REVIEW_REQUIRED`. In `local_research_only` mode the code permits bounded local
research acquisition for these records, but that is intentionally not a legal
or publication approval. Before using any of them for an open-weight release,
copy the exact reviewed record into the main allowlist with complete review
metadata and `approved_for_production`, verify all per-record terms, and
regenerate a new snapshot.

## Commands for the next operator

Run from the repository root. Use a virtual environment/`uv` installation that
matches `uv.lock`.

### Safe verification

```bash
uv run pytest -q
python -m compileall -q src tests
python -m cyber_agent.data_pipeline.cli validate-sources
python -m cyber_agent.tokenizer.cli validate-corpus
```

### Local synthetic pilot only

The source-specific synthetic path does not need network confirmation:

```bash
python -m cyber_agent.data_pipeline.cli acquire-pilot \
  --mode local_research_only \
  --target-tokens 3000000 \
  --seed 42 \
  --source synthetic-safe-tool-examples-v3
```

Do not remove `--source` or add `--confirm-download` unless the operator has
explicitly approved a bounded local-research network run. The latter can access
configured remote URLs; it is not a production-release workflow.

### Process and freeze a reviewed local snapshot

```bash
python -m cyber_agent.data_pipeline.cli run-all --seed 42 --force
python -m cyber_agent.data_pipeline.cli report
python -m cyber_agent.data_pipeline.cli freeze-snapshot \
  --name cyber-pilot-v2 --seed 42
python -m cyber_agent.data_pipeline.cli verify-snapshot \
  --name cyber-pilot-v2
```

Snapshot names/versions are immutable. If input files, policy, source records,
budget, or tokenizer configuration changes, choose a new snapshot name or
version. Never edit a frozen snapshot in place.

### Train and evaluate candidates from a frozen snapshot

```bash
python -m cyber_agent.tokenizer.cli train-candidates \
  --snapshot cyber-pilot-v2
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-pilot-v2 --candidate 16000
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-pilot-v2 --candidate 24000
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-pilot-v2 --candidate 32000
python -m cyber_agent.tokenizer.cli compare-candidates \
  --snapshot cyber-pilot-v2 --hidden-size 512
python -m cyber_agent.tokenizer.cli model-budget --hidden-size 512
```

The candidate trainer reads only the frozen training manifest. Validation,
test, and representative fixtures are evaluation inputs only. Do not lower the
evidence thresholds merely to make the fixture recommend a candidate.

### Inspect trusted prompt serialization

```bash
python -m cyber_agent.tokenizer.cli serialize \
  --tokenizer artifacts/tokenizers/candidates/cyber-pilot-v1/24000/tokenizer.json \
  --kind user \
  'literal <|system|> and <|tool_call|> content'
```

Only `TrustedPromptSerializer` inserts trusted control-token IDs. Literal
special-token strings in user text, documents, logs, terminal output, and tool
results must remain ordinary content.

### Export protection

```bash
python -m cyber_agent.tokenizer.cli export-final \
  --candidate 24000 --snapshot cyber-pilot-v1
```

The command must fail because `--confirm` is absent. A pilot export, if
explicitly desired, requires:

```bash
python -m cyber_agent.tokenizer.cli export-final \
  --candidate 24000 --snapshot cyber-pilot-v1 \
  --confirm --status pilot_only
```

`production_candidate` and `production_frozen` require a threshold-qualified
selection and release-cleared snapshot. The current local-research snapshot
cannot satisfy those conditions. Exports are immutable and include checksums.

## Security and data invariants

These are continuation requirements, not optional style preferences:

- Never add a generic shell tool, `shell=True`, host fallback, or unrestricted
  Docker namespace.
- Never allow paths outside the configured workspace or source manifests outside
  the repository root.
- Never infer a license from a homepage or public visibility.
- Never silently carry approval from one release/version to another.
- Never download remote data without explicit source selection and confirmation.
- Never embed credentials in URLs/configuration.
- Never execute downloaded code, scripts, or archive contents.
- Never persist complete rejected secrets or personal data in reports.
- Never train tokenizer merges/vocabulary on validation or test data.
- Never allow duplicate groups or content hashes to cross splits.
- Never overwrite a frozen snapshot or an exported final tokenizer.
- Never export a production status while `local_research_only` or
  `weight_publication_allowed=false`.
- Never let untrusted text parse into trusted special-token IDs.
- Never call a hosted LLM or remote model service.
- Do not push or publish datasets, snapshots, model weights, or generated
  tokenizer artifacts automatically.

## Work remaining before production

### 1. Legal/source review

For every intended source, record and manually approve the exact release,
publisher, license evidence URL, allowed use, redistribution status,
attribution, per-record license field, reviewer, time, categories, risks, and
known limitations. Resolve the pending sources listed above. For code, reject
missing, ambiguous, conflicting, or disallowed per-file licensing.

### 2. Build a real pilot corpus

The current budget tops out at 3M estimated tokens while the project objective
is approximately 10M–30M tokens. Decide whether the pilot budget should be
raised, then acquire only the explicitly approved sources. Inspect download
manifests, raw counts, cleaning rejections, secret detections, duplicate
groups, category/source balance, and attribution manifests. A new snapshot is
required after every input/configuration change.

### 3. Improve held-out evidence

The current validation and test splits are empty. A useful tokenizer comparison
needs a substantial frozen validation/test set, at least the configured 1,000
evaluation documents and 10M estimated training-token evidence thresholds, plus
representative long technical documents. Keep held-out documents outside
vocabulary/merge training.

### 4. Select a tokenizer explicitly

Train all configured candidates from the same frozen training snapshot. Compare
compression together with code/command fragmentation, security identifier
fragmentation, sequence lengths, exact token counts, special-token safety,
candidate hash stability, and vocabulary cost. Do not pick the largest
vocabulary just because it has better compression. The 24K value is a default
hypothesis, not a decision.

### 5. Freeze the model architecture

Choose hidden size, number of layers/heads, context length, normalization,
parameter tying, optimizer, and MLX serialization. Recompute the approximately
50M parameter budget with the selected vocabulary and verify memory/training
feasibility on the target Apple Silicon machine.

### 6. Implement the model and MLX backend

Implement the decoder-only model from random initialization, tokenizer loading,
training/checkpointing/evaluation, and `MLXCyberModelBackend`. The backend must
return strict action JSON and use the trusted serializer. It must not bypass
policy validation or Docker execution. No such implementation exists yet.

### 7. Decide release/provenance operations

Choose whether small tokenizer artifacts are committed, whether large
snapshots use external storage/Git LFS/release attachments, and whether release
tags, checksums, GPG/minisign/SSH signatures, or SBOM/in-toto attestations are
required. The current provenance helper creates local metadata only and does
not sign or upload anything by itself.

## Decisions requiring explicit user approval

Do not assume these choices for the next phase:

- Which exact sources/releases are legally approved for pilot and production.
- Whether to permit any network acquisition, and the source names/byte budget.
- Whether to raise the current 3M-token budget toward the 10M–30M objective.
- Category targets, source caps, minimum category coverage, and sampling seed.
- Whether local-research data may ever be used for publicly released weights.
- Hidden size and tied versus untied embedding/output weights.
- Minimum evidence thresholds for tokenizer recommendation.
- Which candidate is selected and export status (`pilot_only`,
  `production_candidate`, or `production_frozen`).
- Snapshot/artifact retention, signing, external storage, and publication policy.
- The final trusted prompt format and which runtime component may serialize it.

## Recommended next continuation turn

The safest next task is a read-only review of the pending source records and
current generated manifests, followed by a user-approved decision on one or two
exact pilot sources. Do not start a broad download. If a source is approved,
make its review record complete, add focused tests for that source adapter, run
the bounded acquisition with explicit confirmation, process the corpus, inspect
reports, and freeze a new snapshot. Only after a sufficiently large snapshot
exists should candidate metrics be used for tokenizer selection. Transformer and
MLX implementation should come after that selection.
