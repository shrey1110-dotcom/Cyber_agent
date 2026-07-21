# cyber-agent — Phase 1 safe agent shell

`cyber-agent` is the policy and sandbox layer for a future, locally trained
decoder-only cybersecurity model. Phase 1 deliberately contains no language
model weights, MLX training code, pretrained model, hosted-model SDK, or
external model API call. The included deterministic backend is temporary test
infrastructure.

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

## Connecting the custom MLX model later

The agent depends only on this protocol:

```python
class ModelBackend(Protocol):
    def generate(self, messages: list[Message]) -> str:
        ...
```

Phase 2 can implement the protocol and swap only the construction site:

```python
backend = MLXCyberModelBackend(
    model_path="models/cyber-50m/final"
)
agent = Agent(backend=backend, tools=registry, logger=logger)
```

`MLXCyberModelBackend.generate` will serialize the conversation into the
custom model's chat/prompt format, run local MLX inference, and return exactly
one action JSON string. The parser, policy, Docker runtime, tool-result format,
and loop do not depend on MLX and need no redesign. The eventual backend must
use the model trained from random initialization; the temporary deterministic
backend must not be shipped as the final model.

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
- Phase 1 does not yet include model-side constrained decoding, prompt-injection
  defenses, MLX inference, model training, authentication, or multi-user
  authorization.
