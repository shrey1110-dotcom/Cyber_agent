# Phase 4 local MLX pretraining

This package starts the model-training implementation. It builds a decoder-only
transformer from random initialization on Apple Silicon using MLX. It does not
download a pretrained model, use an external LLM, or call a hosted model API.

## Target architecture

`config/training.json` describes `cyber-decoder-v1`: 11 decoder blocks, hidden
size 512, eight attention heads, a SwiGLU MLP of width 1536, RMSNorm, learned
position embeddings, and tied 24K input/output embeddings. The exact parameter
count is emitted by `inspect-model`; it is designed to be approximately 50M,
not an assertion that a useful 50M model has already been trained.

The model consumes integer IDs from the custom byte-level BPE tokenizer. It
uses teacher-forced next-token cross entropy, AdamW, clipping, a deterministic
learning-rate schedule, immutable checkpoints, and held-out validation.

## Data and release gates

Training reads only `train_manifest.jsonl` from an immutable snapshot. The
loader verifies the snapshot checksums, candidate tokenizer hashes, matching
snapshot hashes, stable vocabulary size, and split isolation before it creates
a token block. Validation and test manifests are never used for vocabulary
learning or training batches.

The current `cyber-pilot-v7` snapshot and its candidates are
`local_research_only` / `pilot_only`. A full run therefore requires both
`--allow-pilot-artifacts` and `--confirm-start`; its checkpoints are marked
`local_research_pilot` and are not publishable. A tiny smoke run is also marked
non-production. Production training remains gated on a larger reviewed corpus
and an explicit tokenizer selection/export.

`cyber-pilot-v7` contains 4,679 retained documents and 4,081,394 provisional
tokens. It is useful for exercising the complete local training path but does
not meet the configured 10M-token / 1,000-held-out-document evidence gates.
The `pilot-v7-50m-bootstrap-v1` run completed exactly one 2,048-token update
on the target 50,048,512-parameter architecture; it is a mechanics check, not
a quality or capability claim.

## Commands

Inspect the target without writing weights:

```bash
python -m cyber_agent.training.cli inspect-model \
  --tokenizer artifacts/tokenizers/candidates/cyber-pilot-v7/24000/tokenizer.json
```

Verify the mechanics with a small, two-layer random model (not a 50M training
run):

```bash
python -m cyber_agent.training.cli smoke-train \
  --snapshot cyber-pilot-v7 \
  --tokenizer artifacts/tokenizers/candidates/cyber-pilot-v7/24000/tokenizer.json \
  --run-name pilot-smoke-v1 --steps 2
```

Starting the target architecture is an explicit, local operation:

```bash
python -m cyber_agent.training.cli train \
  --snapshot cyber-pilot-v7 \
  --tokenizer artifacts/tokenizers/candidates/cyber-pilot-v7/24000/tokenizer.json \
  --run-name pilot-50m-v1 \
  --allow-pilot-artifacts --confirm-start
```

For a deliberately bounded architecture smoke step, add `--max-steps 1
--evaluation-max-batches 1`. This verifies the full target shape without
claiming that one step creates a usable model.

All runs remain local. Checkpoints land below `artifacts/training/`, are
Git-ignored, and contain `model.safetensors`, `optimizer.safetensors`, exact
data/tokenizer provenance, metrics, and SHA-256 checksums. Existing checkpoint
directories are immutable; resuming verifies model, data, tokenizer, and
configuration compatibility before loading any arrays.

`compile_steps` remains disabled in the checked-in configuration. Compile and
memory/performance tuning are deferred until the production context length,
batch size, and hardware budget are measured; a future implementation must
record any compiled execution configuration in the immutable run manifest.

## Later agent integration

The training package deliberately does not implement inference serving or
attach weights to `ModelBackend`. Once a production-frozen tokenizer/dataset
and a validated checkpoint exist, a later `MLXCyberModelBackend` can load this
model, use the trusted serializer from `cyber_agent.tokenizer.serialization`,
and return the already-defined structured action JSON. Runtime tool policy and
Docker isolation remain outside the model and are still enforced.
