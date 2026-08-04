# 3B-token local-research corpus plan

## Training gate

A future 150M-parameter model is **not** allowed to start final pretraining
from the current 145.5M-token pilot. The target is a separately frozen,
explicitly weighted mixture of at least 3B *unique, post-filtering* tokenizer
tokens. This is a compute/data planning target, not a claim that every model
requires exactly this ratio.

The mixture objective is configurable:

| Category | Target share | Initial token goal |
| --- | ---: | ---: |
| General technical English | 30% | 900M |
| Permissively licensed code | 25% | 750M |
| Linux / system administration | 15% | 450M |
| Networking | 10% | 300M |
| Cybersecurity documentation | 15% | 450M |
| Structured terminal/tool examples | 5% | 150M |

No collection may be counted twice. In particular,
`research-150m-v1` is an older subset of `research-general-v2` and is excluded
from the planned mixture.

## First large tranche: English Wikipedia

`research-3b-general-v1` is a bounded, isolated acquisition of up to 920M
**provisional** tokens from the exact `20260701-pages-articles-multistream`
English Wikipedia dump. The upstream advertises a 26,564,488,717-byte file;
the collection hard caps transfers at 28GB, reads the bzip2 XML stream without
extracting archives or executing content, and stops materialization at 920M
estimated tokens or 1.8M source documents.

The official dump listing supplies only the legacy SHA-1 value
`736667da4f6cd0f2e2ea5ae673a293774ad24749`. The downloader verifies that value
for release fidelity, records an independent SHA-256 in the download manifest,
and uses SHA-256 for all pipeline and snapshot identities.

The source is CC-BY-SA-4.0 and is approved **only** for the project's bounded
private local-research mode. It retains article URL, title, page ID, source
release, attribution requirements, and license metadata. It does not authorize
dataset redistribution, public weights, commercial use, or a claim that the
source is legally cleared for any of those purposes.

## What remains before data is sufficient

This is the general-English tranche only. It cannot meet the 3B-token training
gate on its own, even if it reaches its 920M cap. The next collection versions
must add source-specific, independently reviewed technical sources, especially
code and Q&A/documentation, then deduplicate **across the final mixture** before
train/validation/test splits are made.

The planned Stack Exchange route remains disabled: its source-specific adapter
must preserve post/revision license version, canonical URL, author attribution,
site identity, and per-record provenance before its exact archival release can
be reviewed and enabled. Public accessibility is not approval.

## Commands

Validation does not transfer data:

```bash
uv run python -m cyber_agent.data_pipeline.cli \
  --collection research-3b-general-v1 validate-sources
```

The following is intentionally explicit and resumable. It is the only command
that starts the bounded general tranche:

```bash
uv run python -m cyber_agent.data_pipeline.cli \
  --collection research-3b-general-v1 acquire-research \
  --target-tokens 920000000 --seed 42 --confirm-download \
  --source enwiki-20260701-general-tranche
```

After it completes, run the regular phase-2 gates and freeze a **new** snapshot.
Do not run balancing/splitting or tokenizer training on this general tranche as
if it were the final 3B-token mixture.
