# `cyber-agent-llm-v0`: local compiled pilot chat

`cyber-agent-llm-v0` is an inspection interface for the completed 1,000-step
pilot checkpoint. It loads only the locally generated `cyber-pilot-v7`
checkpoint, verifies its SHA-256 metadata and tokenizer provenance, and runs a
compiled MLX forward pass on Apple Silicon.

It is deliberately **not** the terminal agent and cannot execute tools, read
files, invoke Docker, make network requests, or turn model text into a tool
call. The existing safe agent continues to use its temporary deterministic
backend until a future, evaluated `MLXCyberModelBackend` is explicitly built.

## Start the local chat

From the repository root:

```bash
uv run cyber-chat-v0 info
uv run cyber-chat-v0 --max-new-tokens 64 chat
```

Use `/reset` to discard the local conversation and `/quit` to exit. For a
single prompt:

```bash
uv run cyber-chat-v0 --max-new-tokens 64 prompt "What is a Linux process?"
```

The default runtime is compiled with `mlx.core.compile` and uses a fixed
`[1, 512]` token input shape. This avoids recompiling for every generated token.
The causal mask ensures right-side padding does not affect the next-token logit.
Pass `--uncompiled` only for a local diagnostic comparison.

## Important limitations

The checkpoint has seen only 2,048,000 tokens over 1,000 pilot updates and was
not instruction tuned. It may emit punctuation, fragments, repetitions, or
incorrect statements. Chat access proves that the checkpoint can be loaded and
sampled locally; it does **not** prove it is a useful assistant. Do not rely on
its responses for security decisions.

Generation is deterministic greedy decoding with a hard maximum of 256 newly
generated tokens. Trusted control-token IDs are inserted solely by
`TrustedPromptSerializer`; literal `<|system|>`, `<|assistant|>`, and tool-token
strings in chat input stay ordinary text.

The v0 loader fails closed unless all of these agree:

- checkpoint checksums;
- the checkpoint's random-initialization provenance;
- the frozen train-only snapshot record;
- the candidate tokenizer hash and vocabulary size;
- the local-research/non-publication status.

## Continue training later

The original v0 run is immutable and ended at `max_steps: 1000`. To continue it
without overwriting history, create a new copy of `config/training.json` where
**only** `max_steps` is increased, then create a new run name:

```bash
cp config/training.json /private/tmp/training-v0-continue.json
# Edit only max_steps, for example from 1000 to 2000.
uv run cyber-train --config /private/tmp/training-v0-continue.json train \
  --snapshot cyber-pilot-v7 \
  --tokenizer artifacts/tokenizers/candidates/cyber-pilot-v7/24000/tokenizer.json \
  --run-name pilot-v7-50m-v0-continued-v1 \
  --resume artifacts/training/pilot-v7-50m-bootstrap-v1/checkpoints/step-00001000 \
  --allow-horizon-extension --allow-pilot-artifacts --confirm-start
```

The continuation gate accepts only the same model, tokenizer, frozen data,
optimizer settings, seed, and checkpoint location with a strictly higher
`max_steps`; it records the parent checkpoint in the new immutable run manifest.
This continues a private pilot—not a production or releasable model.
