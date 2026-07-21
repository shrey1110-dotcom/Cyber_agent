# Phase 3.5 reviewed pilot corpus and tokenizer candidates

Phase 3.5 adds bounded acquisition, deterministic balancing, immutable dataset
snapshots, and production-size tokenizer *requests*. It does not approve an
external source, download a large corpus, train a transformer, call a hosted
LLM, or publish data.

## Review gate

Every approved source entry in `config/approved_sources.json` must be pinned to
one exact release and records its publisher, download location, license evidence,
allowed use, redistribution status, attribution, per-record license field,
reviewer, review time, categories, risks, and notes. Only
`approved_for_pilot` and `approved_for_production` permit acquisition or
ingestion. `enabled` is a second independent switch: both are required.

Approval is release-specific. `UNPINNED` cannot be approved, and a manifest
record whose `source_release` differs from the reviewed release is rejected.
Public accessibility is never treated as a license. Code records retain
repository, revision, path, and unambiguous per-record license metadata;
missing, conflicting, unknown, review-required, or denied licensing fails
closed.

The synthetic `sample` release is the only approved fixture source. FineWeb-
Edu, NIST, MITRE ATT&CK, CWE, Linux, Bash, Python, Git, and public-code
placeholders remain pending and disabled. Their entries deliberately do not
claim that a floating homepage establishes reuse permission.

## Acquisition controls

`acquire --source NAME` accepts one exact configured name. The HTTPS URL and
redirect domains come from its reviewed record. The downloader rejects embedded
credentials and arbitrary URLs, uses an identifying user agent, identity
encoding, timeouts, bounded retries, byte budgets, content-length checks,
resumable `.part` files, optional published SHA-256, rate limiting, `fsync`, and
atomic rename. Download manifests record release, URLs, bytes, checksum, resume
offset, attempts, and completion time.

ZIP and TAR extraction rejects absolute paths, `..`, backslash traversal,
symbolic/hard links, devices, and special files. It limits file count, declared
and observed expanded bytes, and compression ratio. Extraction is assembled
under the intended data directory and atomically published. Downloaded programs
and included scripts are never run. No remote source is currently approved, so
the checked-in configuration cannot perform a network download.

## Pilot budget and balancing

`config/pilot_budget.json` contains maximum download bytes, raw and clean
document limits, estimated-token ceiling, source caps, category minimums and
maximums, and configurable 30/25/15/10/15/5 targets. These percentages are
configuration, not hard-coded assumptions.

Before a production tokenizer exists, tokens are provisionally estimated as:

```text
max(1, ceil(UTF-8 bytes / 4))
```

This is not an exact token count. Candidate reports later store exact counts
next to the frozen estimate.

Balancing operates on complete duplicate groups, uses a seeded SHA-256
priority, applies source/category caps before splitting, never duplicates a
document, and records every exclusion and reason. Reports contain raw/final
documents and tokens by source/category, license and rejection distributions,
deduplication impact, sensitive-data counts, source concentration, unmet
minimums, and the limit that ended collection.

## Immutable snapshots

```bash
python -m cyber_agent.data_pipeline.cli freeze-snapshot \
  --name cyber-pilot-v1 --seed 42
python -m cyber_agent.data_pipeline.cli verify-snapshot \
  --name cyber-pilot-v1
```

The snapshot contains all required manifests/reports and SHA-256 checksums. Its
manifest records name/version, timestamp, Git commit, configuration/policy
hashes, seed, input/output hashes, counts, estimate, distributions, limitations,
and readiness. Existing snapshot name/version pairs are never replaced; changed
inputs or configuration require a new name or an incremented `--version`.
Version 1 uses the snapshot name as its directory, while later versions use a
`.vN` suffix (for example, `--name cyber-pilot-v1 --version 2` creates
`cyber-pilot-v1.v2`).

Generated snapshots are ignored by Git. Small final tokenizer artifacts may be
committed directly; larger snapshots may use operator-controlled external
storage, Git LFS, release attachments, signed release tags, or SBOM/in-toto
attestations. Nothing is uploaded automatically and no paid service is required.
`attest-snapshot` creates an adjacent local provenance statement and may
reference an operator-created GPG, minisign, or SSH detached signature.

## Candidate workflow

```bash
python -m cyber_agent.tokenizer.cli train-candidates --snapshot cyber-pilot-v1
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-pilot-v1 --candidate 16000
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-pilot-v1 --candidate 24000
python -m cyber_agent.tokenizer.cli evaluate-candidate \
  --snapshot cyber-pilot-v1 --candidate 32000
python -m cyber_agent.tokenizer.cli compare-candidates \
  --snapshot cyber-pilot-v1 --hidden-size 512
python -m cyber_agent.tokenizer.cli model-budget --hidden-size 512
```

All candidates learn only from the exact frozen training manifest and record
its hash. Validation/test and representative fixtures are evaluation-only.
Reports include requested and actual vocabulary size, exact token counts,
compression, source/category/language metrics, sequence percentiles, round
trips, unknown rate, fragmentation, inspections, coverage, special-token
behavior, and model-budget estimates. A small corpus may produce fewer tokens
than requested; that is reported rather than disguised.

Comparison considers compression, command/code and identifier fragmentation,
parameter cost, corpus size, stability, round trips, special tokens, and the
intended approximately 50M-parameter architecture. It recommends nothing until
all minimum-evidence thresholds pass, every candidate has the same frozen input
and special-token contract, literal controls remain untrusted content, and all
included sources are approved for production. The fixture remains
`insufficient_evidence`.

Export is a separate confirmed operation:

```bash
python -m cyber_agent.tokenizer.cli export-final \
  --candidate 24000 --snapshot cyber-pilot-v1 --confirm
```

Without `--confirm` it fails. The default is `pilot_only`.
`production_candidate` or `production_frozen` additionally requires a
threshold-qualified recommendation. Exports are immutable and contain all
required tokenizer, training, evaluation, selection, vocabulary, and checksum
artifacts.

## Trusted prompt serialization

`TrustedPromptSerializer` alone inserts control-token IDs. Canonical format
`cyber-agent-trusted-prompt-v1` supports system, user, assistant, tool call,
tool result, terminal output, retrieved document, and code block components.
After each trusted boundary, component content is encoded with special parsing
disabled. Literal `<|system|>`, `<|tool_call|>`, and `<|assistant|>` strings in
user text, documents, logs, terminal output, or tool results remain ordinary
bytes.

## Fixture result and production blockers

The fixture snapshot contains four retained documents and approximately 404
provisional tokens. The 16K/24K/32K requests each correctly produce only 458
tokens. Round-trip accuracy is 100 percent with zero unknowns on fixtures, but
the candidates are mechanically identical and cannot support production
selection.

Production still requires explicit approval of exact source releases, legal
evidence and redistribution/attribution handling, budgets and targets, a real
10M-30M-token snapshot, hidden size and tied/untied embeddings, evidence
thresholds, candidate selection, export status, storage/retention, and signing.
