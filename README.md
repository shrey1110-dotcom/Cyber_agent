# cyber-agent — safe agent, data, and tokenizer foundations

`cyber-agent` is the policy and sandbox layer, auditable data pipeline,
tokenizer tooling, and local MLX pretraining foundation for a custom
decoder-only cybersecurity model. It has no pretrained weights, hosted-model
SDK, or external model API call. The included deterministic backend is
temporary test infrastructure.

The default business security floor is maintained in
[`SECURITY.md`](SECURITY.md). Future phases may add explicit, granular,
revocable user-granted permissions—including selected file access—without
allowing the model to expand its own privileges.

Phase 2 adds a local, reproducible, legally auditable training-data pipeline.
See [the data-pipeline guide](docs/data_pipeline.md) for source approval,
licensing, cleaning, secret filtering, deduplication, splitting, resumability,
commands, and unresolved assumptions. It exports untokenized JSONL only.

Phase 3 adds deterministic, train-split-only byte-level BPE tokenizer training,
evaluation, comparison, loading, and explicit final export. See
[the tokenizer guide](docs/tokenizer.md). The checked-in configuration defaults
to a 24K vocabulary and supports 16K/24K/32K production candidates, but the tiny
fixture corpus must use an explicitly labeled small tokenizer. It does not
include pretrained weights, remote dataset downloads by default, or hosted LLM
integration.

Phase 4 implements the local-only MLX `cyber-decoder-v1` training path: a
randomly initialized, tied-embedding decoder-only transformer with 50,048,512
parameters at the 24K pilot vocabulary. It reads only a frozen snapshot's
training split, validates artifact provenance, checkpoints atomically, and
evaluates only held-out splits. The current checkpoint is a private,
one-step research bootstrap—not a useful or releasable model. See
[the training guide](docs/training.md).

`cyber-agent-llm-v0` is a compiled local chat interface for inspecting that
pilot checkpoint. It is intentionally disconnected from tools, Docker, and the
network, and its un-instruction-tuned output may be low quality. See
[the v0 chat guide](docs/chat_v0.md).

For a continuation-oriented description of the current implementation, local
generated state, file responsibilities, invariants, and remaining production
work, read the [engineering handoff](docs/HANDOFF.md).

Phase 3.5 adds exact-release source reviews, bounded resumable acquisition,
safe archives, configurable pilot budgets, deterministic balancing, immutable
snapshots, frozen-snapshot tokenizer candidates, model-budget analysis,
protected export, checksums/attestations, and injection-safe trusted prompt
serialization. See [the pilot corpus guide](docs/pilot_corpus.md). A bounded
local-research pilot has acquired ten exact reviewed-for-pilot releases; all
other configured sources remain pending/disabled. The resulting snapshot and
tokenizer candidates remain `pilot_only`, are ignored by Git, and are not
cleared for redistribution or released weights.

The local research pilot milestone adds a separately labeled private-data mode,
conservative 1M–3M-token budgets, configured official/source archives,
deterministic safe tool examples, source materializers, balancing before
splitting, publication blocking, manual accepted-sample inspection, and
train-only tokenizer pilot aliases. See
[the local research pilot guide](docs/local_research_pilot.md). Network corpus
acquisition requires `--confirm-download`; downloaded data and generated
artifacts are never committed or pushed.

The implemented flow is:

```text
user request
  -> ModelBackend.generate(messages)
  -> strict tool_call JSON parsing
  -> closed tool registry + argument/path policy
  -> fresh non-root, network-disabled Docker container
  -> strict tool_result JSON
  -> ModelBackend.generate(messages)
  -> final_answer JSON or another bounded tool call
```

## The five tools

These are the only registered operations:

| Tool | Arguments | Behavior |
| --- | --- | --- |
| `list_files` | `{"path": ".", "recursive": false}` | Lists at most 2,000 entries below a workspace directory. |
| `read_file` | `{"path": "README.md"}` | Reads at most 1 MiB from one workspace file. |
| `check_processes` | `{}` | Reads the isolated container's `/proc` process view. |
| `check_ports` | `{}` | Reads listening TCP sockets in the container network namespace. |
| `run_tests` | `{"path": "."}` | Runs one fixed `python -m pytest ...` argv in a workspace directory. No flags or commands are model-controlled. |

There is no generic command tool. Every argument object uses a closed schema,
and both subprocess sites use argument arrays with `shell=False`.

## Setup and tests

Python 3.11 or newer and Docker are required.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest
docker build -t cyber-agent-sandbox:latest .
```

The unit tests do not require a Docker daemon: they inspect the exact Docker
invocation through an injected subprocess boundary. Actual tool execution does
require the image and a running Docker daemon; production code intentionally
has no host-execution fallback.

## Deterministic demonstrations

From this directory, after building the image:

```bash
cyber-agent "Read the documentation" --workspace "$PWD" --demo-path README.md
cyber-agent "Try to escape the workspace" --workspace "$PWD" --traversal-demo
```

The first request executes `read_file` inside Docker. The second is rejected by
host policy before Docker starts. Audit events are JSON lines on stderr by
default; pass `--log-file audit.jsonl` to retain them.

The model action schema is:

```json
{
  "type": "tool_call",
  "tool": "read_file",
  "arguments": {"path": "README.md"},
  "reason": "Read the project documentation."
}
```

Tool results sent back to the model are:

```json
{
  "type": "tool_result",
  "tool": "read_file",
  "status": "success",
  "output": "file contents",
  "error": null
}
```

Final answers must be:

```json
{"type": "final_answer", "content": "The requested explanation."}
```

Unknown keys, unknown action types, unknown tools, and wrong argument types are
rejected.

## Connecting the trained custom MLX model later

The agent depends only on this protocol:

```python
class ModelBackend(Protocol):
    def generate(self, messages: list[Message]) -> str:
        ...
```

A later model phase can implement the protocol and swap only the construction
site:

```python
backend = MLXCyberModelBackend(
    model_path="models/cyber-50m/final"
)
agent = Agent(backend=backend, tools=registry, logger=logger)
```

`MLXCyberModelBackend.generate` remains a future inference integration. It will
load a production-frozen Phase 4 checkpoint and final tokenizer, serialize the
conversation into the custom model's chat/prompt format, run local MLX
inference, and return exactly one action JSON string. The parser, policy,
Docker runtime, tool-result format, and loop do not depend on MLX and need no
redesign. It must use the random-initialized model trained locally; the
temporary deterministic backend must not be shipped as the final model.

## Security controls

- A closed registry enforces exactly five named tools and closed argument sets.
- Paths are checked for null bytes, traversal components, foreign absolute
  roots, and backslashes. Canonical resolution rejects symlink escapes.
- File paths are validated once on the host and again in the container.
- The workspace bind mount is read-only and appears only at `/workspace`.
- Each call uses a fresh container, UID/GID `10001`, no capabilities,
  `no-new-privileges`, a read-only root filesystem, a constrained temporary
  filesystem, and CPU, memory, PID, and wall-clock limits.
- Timed-out containers are force-removed by their unique per-call name as a
  best-effort cleanup in addition to Docker's `--rm` lifecycle flag.
- `--network none` is mandatory in the Docker argv.
- Tool output and directory listing sizes are bounded.
- Requests, policy rejections, executions, failures, and completions are logged
  as structured JSON events.
- The agent limits chained tool calls, and all model output is strict JSON.

## Remaining limitations

- Docker is a security boundary with a nonzero attack surface; keep the daemon,
  base image, and host OS patched. For stronger isolation, add a hardened
  runtime such as gVisor or a dedicated VM in deployment.
- A read-only mount prevents container writes but not concurrent host changes.
  Revalidation inside the container narrows, but cannot mathematically remove,
  every filesystem race.
- `run_tests` executes repository test code. It is confined to the container,
  but malicious tests can still consume resources up to the configured limits
  and inspect files included in the mounted workspace.
- Process and port checks intentionally describe the isolated container, not
  the Docker host. Granting host PID or network namespaces would weaken the
  boundary and is not supported.
- Logs contain user requests and tool arguments. Protect and rotate retained
  audit files; file contents and tool output are not included in audit events.
- Docker's `--network none` still leaves the loopback interface available.
- The project has local pretraining mechanics but not MLX inference,
  model-side constrained decoding, capability evaluation, instruction tuning,
  authentication, or multi-user authorization.
- The current pilot corpus/tokenizers/checkpoints are explicitly private
  research artifacts and lack the corpus size, legal release clearance, and
  evaluation evidence required for a production model.
