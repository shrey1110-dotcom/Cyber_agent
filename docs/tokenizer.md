# Phase 3 tokenizer pipeline

Phase 3 converts approved Phase 2 training text into a deterministic token
vocabulary. A tokenizer maps text to integer IDs consumed by a future model and
maps generated IDs back to text. This phase trains no transformer, downloads no
dataset or model, and calls no hosted service.

## Why byte-level BPE

Byte-pair encoding begins with byte-level coverage and repeatedly learns useful
adjacent byte combinations. It can therefore represent arbitrary valid UTF-8,
including uncommon security identifiers and Unicode, while learning compact
pieces for frequent prose, code, commands, paths, and logs. The byte alphabet
also keeps the normal-text unknown-token rate at zero in the expected case.

Training uses the local `tokenizers` package, a BPE model, a byte-level
pre-tokenizer and decoder, byte fallback, and an initial byte alphabet. It uses
no tokenizer-side normalization: capitalization, punctuation, indentation,
line breaks, shell operators, command flags, hashes, URLs, IP addresses, paths,
and identifiers are preserved. Phase 2 has already validated UTF-8 and applied
NFC Unicode normalization before splitting; Phase 3 does not normalize those
outputs again.

Code whitespace is data. Collapsing indentation or line breaks can change
Python blocks, YAML structure, shell continuations, logs, and the exact JSON that
the agent must emit, so corpus streaming and tokenization preserve it.

## Train-only and licensing gate

The tokenizer accepts exactly `data/splits/train.jsonl`. It fails closed if the
Phase 2 train, validation, test, source, or split manifests are missing or do not
match their files. Each selected training record must have a valid source,
source URL, approved license, category, language, content hash, and nonempty
text. Document IDs found in validation or test are rejected, and evaluation
reads only the representative fixture plus Phase 2 validation and test splits.
Phase 2 outputs are read-only inputs.

Selection is deterministic for a fixed configuration and seed. Configurable
per-source limits prevent one source from dominating; category limits and
weights support category-aware deterministic sampling. Text is streamed from
the selected records rather than accumulated into one in-memory corpus. The
training manifest records every selected document ID and content hash, input
manifest hashes, source/category/license counts, configuration hash, seed,
package versions, sizes, and artifact hashes.

## Reserved control tokens

The order in `config/tokenizer.json` is an ID contract. Configuration validation
rejects reordering, and training verifies the resulting IDs.

| ID | Token | Purpose |
| ---: | --- | --- |
| 0 | `<|pad|>` | Batch padding; attention mask value is zero. |
| 1 | `<|bos|>` | Beginning of a sequence. |
| 2 | `<|eos|>` | End of a sequence or completed answer. |
| 3 | `<|unk|>` | Defensive fallback; byte coverage should avoid it for normal text. |
| 4 | `<|system|>` | Trusted system-message boundary. |
| 5 | `<|user|>` | User-message boundary. |
| 6 | `<|assistant|>` | Assistant-message boundary. |
| 7 | `<|tool_call|>` | Structured tool-call boundary. |
| 8 | `<|tool_result|>` | Structured tool-result boundary. |
| 9 | `<|terminal|>` | Terminal text or output boundary. |
| 10 | `<|document|>` | Document-content boundary. |
| 11 | `<|code|>` | Source-code boundary. |

Normal encoding treats a string such as `<|system|>` as literal text and does
not turn it into ID 4. Trusted prompt construction must opt in with
`parse_special_tokens=True`. This prevents untrusted document or user text from
accidentally becoming a control boundary.

## Configuration and vocabulary size

The strongly typed configuration is in `config/tokenizer.json`. The production
default is 24,000 tokens, with 16K, 24K, and 32K comparison candidates. Smaller
vocabularies usually create longer sequences but leave more of a 50M-parameter
budget for transformer layers. Input embeddings use approximately
`vocabulary_size x hidden_size` parameters; an untied output projection can add
the same cost again. Larger vocabularies may compress representative text more
efficiently, but they are not automatically better.

Candidate selection must use measured validation/test metrics after the real
approved corpus exists. The comparison command deliberately never selects or
exports a winner. It reports insufficient data unless all configured production
candidates have non-fixture evaluation data of adequate size. The 24K default
is a hypothesis, not a decision.

## Commands

Run from the repository root:

```bash
python -m cyber_agent.tokenizer.cli validate-corpus
python -m cyber_agent.tokenizer.cli train --vocab-size 16000 --seed 42
python -m cyber_agent.tokenizer.cli train --vocab-size 24000 --seed 42
python -m cyber_agent.tokenizer.cli train --vocab-size 32000 --seed 42
python -m cyber_agent.tokenizer.cli evaluate \
  --tokenizer artifacts/tokenizers/candidates/24000/tokenizer.json
python -m cyber_agent.tokenizer.cli compare
python -m cyber_agent.tokenizer.cli inspect \
  --tokenizer artifacts/tokenizers/candidates/24000/tokenizer.json \
  "journalctl -u ssh --since '2 hours ago'"
python -m cyber_agent.tokenizer.cli export-final --candidate 24000
```

`export-final` is the only operation that populates
`artifacts/tokenizers/final`, and an operator must invoke it explicitly for an
already evaluated candidate. Candidate and final artifacts are generated files
and are Git-ignored.

The current four-document fixture is intentionally too small for meaningful
16K, 24K, or 32K training. To verify mechanics only, train a deliberately small,
clearly marked artifact:

```bash
python -m cyber_agent.tokenizer.cli train \
  --vocab-size 512 --seed 42 --fixture
python -m cyber_agent.tokenizer.cli evaluate \
  --tokenizer artifacts/tokenizers/candidates/512/tokenizer.json
```

The candidate contains `FIXTURE_ONLY.txt`, its manifest sets
`fixture_artifact: true` and `production_ready: false`, and its metrics must not
be used for production selection. A tiny corpus cannot populate a requested
large vocabulary or represent the intended domain distribution.

## Artifacts and evaluation

Each candidate contains `tokenizer.json`, resolved `tokenizer_config.json`,
`special_tokens_map.json`, `training_manifest.json`, `evaluation_report.json`,
`vocabulary.txt`, `vocab.json`, and `merges.txt`. Candidate publishing is
transactional: files are assembled in a temporary directory, then atomically
replace the prior candidate only after all core artifacts and hashes exist.

Evaluation covers English prose, Python, Bash, JSON, YAML, logs, CVE/CWE/MITRE
identifiers, paths, IPv4/IPv6 addresses, and Unicode. It reports token counts,
characters and bytes per token, tokens per document, longest sequence,
unknown-token rate, round-trip accuracy, special-token safety, and metrics by
category, source, and evaluation group. The report records its fixture and
validation/test input hashes and explicitly records that it used zero training
documents for evaluation.

Core artifact hashes exclude timestamped manifests, so deterministic reruns can
be compared directly. Creation timestamps remain in audit manifests and are not
claimed to be deterministic.

## Python and future MLX loading

The model-independent loader returns Python lists and does not add framework
tensors:

```python
from cyber_agent.tokenizer import CyberTokenizer

tokenizer = CyberTokenizer.from_file(
    "artifacts/tokenizers/final/tokenizer.json"
)
token_ids = tokenizer.encode(
    "Check which process is listening on port 8080.",
    add_bos=True,
    add_eos=True,
)
text = tokenizer.decode(token_ids)
batch = tokenizer.batch_encode(
    ["CVE-2026-12345", "/var/log/auth.log"],
    padding=True,
)
```

`CyberTokenizer` exposes vocabulary size, all special-token IDs, encoding,
decoding, batch encoding, BOS/EOS insertion, maximum-length truncation, padding,
and attention masks. Future MLX pretraining will load the explicitly exported
final `tokenizer.json`, use its vocabulary size to configure embeddings and
logits, and convert these lists to MLX arrays in the model/training layer.

## Decisions still open before production training

- Complete legal review and freeze a sufficiently large, representative Phase
  2 training snapshot with operational attribution handling.
- Measure 16K, 24K, and 32K candidates on substantial held-out data and select a
  vocabulary jointly with model hidden size, tied/untied output weights, context
  length, and the approximately 50M total parameter budget.
- Tune source/category quotas, minimum frequency, maximum token length, and
  cybersecurity/code balance using audited corpus statistics.
- Revisit trusted serialization only through a versioned format;
  `TrustedPromptSerializer` is now the sole control-token inserter.
- Decide whether a reviewed small final tokenizer is versioned or all generated
  artifacts remain outside Git, and define artifact signing/release retention.

The fixture tokenizer is test infrastructure, not a production language-model
component.

Frozen-snapshot candidates, 50M model-budget estimates, evidence-gated
selection, confirmed export, provenance, and canonical trusted serialization
are documented in [the Phase 3.5 guide](pilot_corpus.md).
