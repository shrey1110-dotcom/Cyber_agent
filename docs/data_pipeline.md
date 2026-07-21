# Phase 2 training-data pipeline

Phase 2 creates legally reviewable, untokenized JSONL for the future model. It
does not train a tokenizer or model and does not contact a hosted model service.
The implementation uses only the Python 3.11 standard library.

## Pipeline stages

1. `validate-sources` checks every configured source against the license policy.
2. `ingest` reads enabled local manifests, validates UTF-8, requires per-record
   license metadata, and rejects sources outside the allowlist.
3. `clean` extracts HTML/plain text, normalizes Unicode, preserves code
   indentation, detects sensitive data, applies hard quality checks, and assigns
   a score.
4. `deduplicate` removes exact SHA-256 duplicates and near duplicates detected
   with 64-bit SimHash plus three-token-shingle similarity.
5. `split` assigns entire duplicate groups deterministically using a seeded
   SHA-256 value and the configured 98/1/1 boundaries.
6. `export` writes JSONL documents and source, rejection, duplicate, and split
   manifests.
7. `report` creates aggregate dataset and license reports without rejected text.

Each completed data stage writes a marker under `data/manifests/stages/`. The
marker records its input fingerprint, outputs, counts, and completion time. A
matching stage is skipped on resume. Outputs use a same-directory temporary file,
`fsync`, and atomic replacement; a marker is written only after every stage
output succeeds.

## Commands

Run from the repository root:

```bash
python -m cyber_agent.data_pipeline.cli validate-sources
python -m cyber_agent.data_pipeline.cli ingest --source sample
python -m cyber_agent.data_pipeline.cli clean
python -m cyber_agent.data_pipeline.cli deduplicate
python -m cyber_agent.data_pipeline.cli split --seed 42
python -m cyber_agent.data_pipeline.cli export
python -m cyber_agent.data_pipeline.cli report
python -m cyber_agent.data_pipeline.cli run-all --seed 42
```

Pass `--force` to a data-changing stage when a reviewed operator deliberately
wants to regenerate it. Generated data is ignored by Git.

## Adding a new approved source

1. Establish the exact dataset snapshot, revision, files, and original URLs.
2. Obtain and manually review the governing license and any file-level licenses.
   Public visibility is not permission. Do not infer or replace a missing license.
3. Add the exact license identifier to `config/license_policy.json`. Set it to
   `review_required` until legal review says the intended training use and
   obligations are acceptable. Record whether attribution is required.
4. Add the full review record to `config/approved_sources.json`: exact release,
   homepage/download location, publisher, license and evidence URL, allowed use,
   redistribution status, attribution, per-record license field, review status,
   reviewer/time, categories, risks, and notes. Keep `enabled` false and status
   `pending` during review.
5. Prepare a small local JSONL manifest. Every line must contain `path`,
   `source_url`, `license`, `retrieved_at`, `language`, `media_type`, and useful
   provenance metadata. Paths must stay below the manifest directory.
6. Implement a reviewed local adapter if `local_manifest` is insufficient. Phase
   2 intentionally has no remote bulk-downloader.
7. Run source validation and the complete fixture/test pipeline. Inspect the
   source, rejection, duplicate, split, license, and summary reports.
8. Enable the source only after the license rule is `allowed`, attribution is
   operationally preserved, and a reviewer signs off on the pinned snapshot.

## Quality score

Accepted documents receive a deterministic score from 0.0 to 1.0. Length, token
diversity, printable/readable characters, English markers, and low markup or
navigation dominance each contribute 20 percent. Hard failures—including empty,
short, corrupt, repetitive, markup-dominated, navigation-dominated, unreadable,
spam, generated-garbage, and non-English records—reject the document even if the
numeric score would otherwise pass. Code skips the prose English-marker test and
is normalized without collapsing meaningful leading whitespace.

## Sensitive-data behavior

The default action is rejection. Pattern findings contain only type and character
offsets. Reports retain the document identifier, source, stage, and safe reason
codes; they do not retain the matched value or rejected document text. A redaction
implementation exists for a future reviewed policy, but the Phase 2 configuration
uses fail-closed rejection.

## Unresolved licensing assumptions

Only the synthetic `sample` fixture is enabled. All external placeholders remain
disabled. Before enabling them, resolve at least these questions:

- FineWeb-Edu: the dataset-level terms do not establish the rights of every
  underlying web document; per-record provenance and license handling are needed.
- NIST: confirm each publication and exclude or separately handle third-party
  figures, standards text, images, and incorporated material.
- MITRE ATT&CK: pin a release and verify its license and notices on that release.
- CWE: review the exact release terms; the configuration intentionally does not
  guess a license.
- Linux documentation: honor file-level SPDX identifiers and authorship because
  licensing can vary.
- Bash and Git documentation: resolve applicable copyleft, attribution, source,
  and distribution obligations.
- Python documentation: pin the release and preserve PSF and third-party notices.
- Public code: verify the license and provenance of every repository/file; a
  public repository is not automatically reusable training data.

This document is an engineering control, not legal advice. The generated summary
repeats unresolved source assumptions so they remain visible during every review.

Phase 3.5 acquisition, budgets, balancing, immutable snapshots, and exact-release
approval are documented in [the pilot corpus guide](pilot_corpus.md).
