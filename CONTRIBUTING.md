# Contributing to Cyber Agent

Cyber Agent treats the model, tool policy, data pipeline, and sandbox as
separate trust boundaries. Keep contributions focused so each boundary can be
reviewed and tested independently.

## Development setup

Cyber Agent requires Python 3.11 or newer. Docker is needed for real sandbox
execution, but the unit tests use an injected subprocess boundary and do not
require a running daemon.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pytest
docker build -t cyber-agent-sandbox:latest .
```

Use synthetic fixtures for tests. Do not commit downloaded corpora, generated
model artifacts, credentials, private data, or retained audit logs.

## Change expectations

- Add or update tests for behavior changes.
- Keep schemas closed and reject unknown fields.
- Preserve fixed argument arrays with `shell=False`.
- Keep the default workspace mount read-only and networking disabled.
- Document changes to data provenance, licensing, filtering, or export rules.
- Update `SECURITY.md` when a change alters a trust boundary or permission.

Guardrail-affecting changes must also complete the change-control checks in
`SECURITY.md`, including a Docker-backed allowed call and rejected boundary
violation.

## Pull requests

Describe:

1. the problem and intended outcome;
2. the trust boundaries affected;
3. the validation performed;
4. any new data, dependency, permission, network, or host-access implications;
5. the rollback or failure behavior.

Keep unrelated refactors separate. For security fixes, avoid publishing
exploit details before a coordinated patch is available; use GitHub's private
security advisory flow instead.
