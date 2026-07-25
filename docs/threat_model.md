# Threat Model

This threat model describes the current local Cyber Agent architecture. It
should be updated whenever a change adds a tool, permission, data source,
network path, model backend, mount, or execution mode.

## Protected assets

- Files within and outside the selected workspace
- Host credentials, environment variables, processes, and network access
- Training data licenses, provenance records, and immutable snapshots
- Private source material and rejected sensitive samples
- Audit records, generated artifacts, and tokenizer exports
- The integrity of the tool registry, policy, and sandbox configuration

## Trust boundaries

1. **User input to model backend:** Requests and attached content may contain
   prompt injection or misleading instructions.
2. **Model output to action parser:** Model text is untrusted until it passes
   the strict JSON action schema.
3. **Action parser to tool policy:** A valid action may still request a
   forbidden tool, argument, path, or capability.
4. **Host policy to container runtime:** Approved calls cross into a fresh
   non-root, network-disabled Docker container with a read-only workspace.
5. **External source to data pipeline:** Acquired content may be malicious,
   incorrectly licensed, duplicated, poisoned, or contain secrets.
6. **Generated artifact to export:** Tokenizers, snapshots, and future model
   artifacts remain untrusted until checksums, reports, and explicit export
   rules pass.

## Primary threats and controls

| Threat | Current control |
| --- | --- |
| Prompt injection requests a privileged action | Closed action schema, closed tool registry, and host policy validation |
| Path traversal or symlink escape | Canonical host validation plus container-side revalidation |
| Arbitrary command execution | No generic command tool; fixed argument arrays with `shell=False` |
| Host or network discovery | Fresh container, separate namespaces, `--network none`, no host socket |
| Container privilege escalation | Non-root UID/GID, dropped capabilities, `no-new-privileges`, read-only root |
| Denial of service | CPU, memory, PID, output, temporary-filesystem, and wall-clock limits |
| Secret or personal-data ingestion | Sensitive-data filtering, rejection reports, local private-data mode, publication blocking |
| Training-data poisoning | Approved-source configuration, provenance, deduplication, quality checks, and immutable snapshots |
| Audit-log disclosure | Raw file contents and tool output excluded by default; retained logs require access control |
| Silent policy downgrade | Fail-closed errors and no host-execution fallback |

## Explicit non-goals

The current phase does not claim to protect against:

- a compromised Docker daemon or host operating system;
- vulnerabilities in the container runtime or kernel;
- malicious repository tests consuming resources up to configured limits;
- every concurrent filesystem race on a host-modified workspace;
- model alignment failures beyond the deterministic parser and policy boundary;
- secure multi-user authorization or remote deployment.

## Security review triggers

A focused security review is required when a change:

- adds or broadens a tool, mount, permission, subprocess, or network path;
- introduces hosted inference, remote data acquisition, or a new external
  service;
- changes path canonicalization, symlink handling, Docker arguments, limits,
  or cleanup;
- alters source approval, licensing, secret filtering, snapshot, or export
  behavior;
- records additional request, file, tool-output, or identity data;
- adds a production model backend or changes the action serialization format.

For these changes, run the change-control checks in `SECURITY.md` and document
the affected assets, attacker capability, failure mode, and rollback plan in
the pull request.
