"""Free, local artifact checksums and optional unsigned/signed attestation metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from cyber_agent.data_pipeline.export import atomic_write_json
from cyber_agent.data_pipeline.schemas import utc_now


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_provenance_attestation(
    directory: Path,
    *,
    artifact_type: str,
    signer_identity: str | None = None,
    detached_signature: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create a local provenance statement; optionally reference a detached signature.

    Cryptographic signing is intentionally delegated to an operator-selected free
    tool (for example ssh-keygen, minisign, or GPG).  This function never invokes
    a signer or uploads an artifact.
    """
    if not directory.is_dir():
        raise ValueError("attestation target must be an artifact directory")
    if not artifact_type.strip():
        raise ValueError("artifact_type is required")
    output = output_path or directory.parent / f"{directory.name}.provenance_attestation.json"
    if output.exists():
        raise ValueError("provenance attestation already exists and will not be overwritten")
    if detached_signature is not None and not detached_signature.is_file():
        raise ValueError("referenced detached signature does not exist")
    subjects = [
        {"name": str(path.relative_to(directory)), "digest": {"sha256": _sha256(path)}}
        for path in sorted(directory.rglob("*"), key=lambda item: str(item.relative_to(directory)))
        if path.is_file()
    ]
    payload: dict[str, Any] = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": "https://cyber-agent.local/provenance/v1",
        "predicate": {
            "artifact_type": artifact_type,
            "generated_at": utc_now(),
            "network_upload_performed": False,
            "signature_status": "detached_signature_referenced" if detached_signature else "unsigned",
            "signer_identity": signer_identity,
            "detached_signature": str(detached_signature) if detached_signature else None,
        },
    }
    atomic_write_json(output, payload)
    return output
