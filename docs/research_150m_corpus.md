# Isolated 150M research-corpus collection

This document defines the next bounded corpus expansion after the small local
pilot. It is a private research collection, not an open-weight release dataset.
Its outputs are isolated from the existing pilot under
`data/collections/research-150m/`; the old pilot manifests, snapshots, and
checkpoints are never overwritten.

## Status and scope

`config/research_budget.json` has a hard ceiling of 150,000,000 provisional
tokens, 1.5 GB of compressed downloads, 850,000 raw documents, 700,000 clean
documents, 80,000,000 estimated tokens per source, and 6.5 GB of decompressed
input per source. Counts before a tokenizer exists are estimates, not exact
training-token counts.

The initial reviewed sources are exact **2026-07-01** Wikimedia dumps:

- Simple English Wikipedia articles: 381,224,225 compressed bytes advertised by
  the dump server when reviewed;
- English Wikibooks articles: 207,098,222 advertised compressed bytes; and
- English Wikiversity articles: 119,592,609 advertised compressed bytes.

They use the declared `CC-BY-SA-4.0` text license. The configuration preserves
the dump URL, per-record article URL/title/page ID, publisher, license evidence,
and detailed attribution requirements. Wikimedia reuse requires attribution,
license notice, and indication of modifications; content may also carry
additional article-specific notices. This project therefore marks the sources
`approved_for_pilot` only for the explicitly authorized private research mode.
It does **not** clear dataset redistribution, commercial use by this project, or
publication of model weights.

## Observed local run: `research-150m-v1`

The first bounded acquisition completed locally on 2026-08-03. It downloaded
707,915,056 bytes, materialized 330,844 source records, and stopped at
149,999,279 provisional source tokens. After UTF-8 validation, secret/PII
filtering, quality checks, exact/near deduplication, and deterministic
balancing, the immutable `research-150m-v1` snapshot contains:

- 210,496 retained documents and 90,000,000 provisional tokens;
- 206,288 train, 2,099 validation, and 2,109 test documents;
- 318 exact plus 16,247 verified near-duplicate removals; and
- 8,163 safe rejection records, including 749 sensitive-data/PII rejections.

The final 90M ceiling is the configured `general` category cap, not a download
failure. The snapshot's checksum-verified content hash is
`6d062600705b7234cfd07edb8b6ccb30c8448e9aa55980a31d24002c36f17e6e`.
It is entirely `general` / `CC-BY-SA-4.0`; it must not be treated as the
balanced final cyber-model corpus or used to justify a production model.

The large run also exercises the bounded-memory implementation: JSONL ingest
and clean stages stream records into temporary directories before atomic
publication; exact deduplication remains exhaustive; and large-corpus
near-duplicate candidates use bounded sampled SimHash bands before full lexical
verification. Balancing uses a distinct hash namespace from splitting so a
source cap cannot bias every selected record into the training split.

The full English Wikipedia dump is intentionally not configured: the dated
2026-07-01 articles dump advertises 26,564,488,717 bytes, well beyond the
bounded 1.5 GB download budget. The Common Pile and The Stack remain unenabled:
their source/per-record rights, terms, and removal processes require a separate
review rather than a blanket approval based on public availability.

## How acquisition is constrained

The `http_wikimedia_xml_bz2` adapter is a data parser, not a general download
or execution mechanism. It accepts only an exact source record with:

- a manually reviewed status and exact release;
- an HTTPS URL and `dumps.wikimedia.org` domain allowlist;
- a source-specific compressed-byte cap and a collection-wide budget;
- timeout, retry, rate-limit, resume, and atomic-download controls; and
- a declared source-level license plus per-record source-page attribution.

The bzip2 XML reader streams pages; it never extracts paths, executes templates,
Lua, JavaScript, or downloaded code. It rejects DTD/entity declarations, limits
decompressed bytes, accepts only main-namespace non-redirect pages, and applies
a deliberately conservative markup cleanup. Normal secret/PII filtering,
quality checks, exact/near deduplication, balancing, and deterministic
splitting run afterwards.

## Commands

Run from the repository root. The source list is explicit so a later source
cannot join the run by accident.

```bash
uv run python -m cyber_agent.data_pipeline.cli \
  --collection research-150m validate-sources

uv run python -m cyber_agent.data_pipeline.cli \
  --collection research-150m acquire-research \
  --target-tokens 150000000 --seed 42 --confirm-download \
  --source simplewiki-20260701-research \
  --source enwikibooks-20260701-research \
  --source enwikiversity-20260701-research

uv run python -m cyber_agent.data_pipeline.cli \
  --collection research-150m run-all --seed 42 --force \
  --source simplewiki-20260701-research \
  --source enwikibooks-20260701-research \
  --source enwikiversity-20260701-research

uv run python -m cyber_agent.data_pipeline.cli \
  --collection research-150m freeze-snapshot \
  --name research-150m-v1 --seed 42
```

The acquisition command requires `--confirm-download` and exits before any
network transfer without it. A failed or interrupted transfer leaves only a
resumable `.part` file; a fully downloaded object and a fully materialized
source directory are published atomically. A completed source is skipped on
rerun. Do not remove the explicit source list or increase the budget just to
make a run succeed.

## What this does not solve

This collection improves general English and educational coverage; it does not
by itself supply the code, Linux, networking, cybersecurity, and tool-use mix
needed for a useful cyber model. The current 50M model needs vastly more than
the existing 4M-token pilot, and even a 150M-token corpus remains an early
research scale rather than a complete 150M-parameter training budget. Before a
new pretraining run, freeze a snapshot, train tokenizer candidates from its
**train** manifest only, compare them on held-out records, then record the
token count and training schedule in a new immutable run manifest.

Future additions require a new exact source review record, a license-policy
entry, per-record provenance sufficient for its source type, a hard source cap,
and a new collection/snapshot version. Do not enable Common Pile, FineWeb,
Dolma, The Stack, random GitHub repositories, or general web crawling without
that work.
