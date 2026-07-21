# Minimum Security Guardrails

These controls are the default security floor for Cyber Agent. They protect the
user unless the user deliberately grants a broader scope. Future features may
add user-controlled permissions, but the model must never grant permissions to
itself or silently weaken the active policy.

## Model boundary

- Production inference remains local. No hosted-model API or external LLM call
  may be introduced into the agent execution path.
- Model output is untrusted input. It must pass the strict action schema before
  it can reach a tool.
- A model cannot create or expand tools, commands, mounts, permissions, or
  timeouts by itself. Only the user-facing permission system may grant a broader
  scope.
- Tool-call loops remain bounded and fail closed on malformed model output.

## Tool boundary

- Phase 1 contains exactly `list_files`, `read_file`, `check_processes`,
  `check_ports`, and `run_tests`. Later tools require an explicit product change
  and an appropriate user permission.
- There is no arbitrary command, shell, script, package-installation, network,
  or host-administration tool.
- Every tool uses a closed JSON argument schema. Unknown tools, unknown fields,
  wrong types, and policy violations are rejected before execution.
- Subprocesses use fixed argument arrays and `shell=False`.

## User-granted permissions

- Users may grant additional permissions, tools, file locations, or container
  capabilities when the product implements a safe permission interface.
- A grant must be explicit, specific, visible before use, attributable to the
  user, and recorded in the audit log. Model text alone is never consent.
- Grants should use the smallest practical scope: a selected file before a
  directory, read-only before read-write, one operation before a session, and a
  session before a persistent grant.
- The interface must clearly distinguish read, write, execute, network, device,
  process, and administrative permissions. High-impact permissions require a
  separate confirmation that names the exact resource and risk.
- Users must be able to inspect and revoke persistent grants. Temporary grants
  expire automatically when the operation or session ends.
- Denial, ambiguity, an unavailable consent interface, or a grant-validation
  failure always leaves the narrower default policy in force.
- A permission grant changes only the named boundary. It does not implicitly
  authorize arbitrary shell execution, unrelated files, additional network
  destinations, or other privilege expansion.

## Filesystem boundary

- Phase 1 file access is confined to `/workspace`. A later permission interface
  may mount user-selected files or directories into dedicated sandbox paths.
- Additional file access requires an exact user-selected path and access mode.
  Read-only is the default; write access requires separate confirmation.
- Null bytes, traversal components, foreign absolute paths, backslashes, missing
  targets, wrong file types, and canonical paths outside the workspace are
  rejected.
- Symlinks are resolved and checked before container launch and again inside the
  container.
- The workspace is mounted read-only in the execution container.

## Container boundary

- Production tools have no host-execution fallback; they run in a fresh Docker
  container.
- Containers run as non-root UID/GID `10001`, with all capabilities dropped,
  `no-new-privileges`, a read-only root filesystem, and no host Docker socket.
- Networking is disabled by default with `--network none`. A future network
  grant must identify its allowed destination and duration; it must not silently
  expose the host network namespace.
- CPU, memory, PID, temporary-filesystem, output-size, and wall-clock limits are
  mandatory. Timed-out containers receive best-effort forced cleanup.
- In Phase 1 the only host mount is the explicitly selected workspace at
  `/workspace`, read-only. Future user-granted mounts remain individually scoped
  and are not treated as general host filesystem access.

## Audit and failure behavior

- Requests, rejections, execution starts, execution failures, cleanups, and
  completions are structured audit events.
- File contents and raw tool output are excluded from audit records by default;
  user requests and tool arguments may be sensitive and retained logs must be
  access-controlled and rotated.
- Parsing, policy, Docker, timeout, and tool errors return structured failures.
  A failure must never fall back to a less isolated execution mode.

## Change-control floor

Before merging a guardrail-affecting change:

1. Run the complete Python 3.11 pytest suite.
2. Build the sandbox image.
3. Run the tests through the Docker-backed `run_tests` tool.
4. Demonstrate one allowed call and one rejected boundary violation.
5. Review the Docker arguments, tool allowlist, path policy, permission-grant
   behavior, and dependency diff.

The temporary deterministic mock backend is test infrastructure only. The
future `MLXCyberModelBackend` must implement the existing `ModelBackend`
interface and remain behind all of these boundaries.
