"""Local inference interfaces for explicitly labeled Cyber Agent model builds.

The package intentionally avoids importing MLX at module import time.  This
keeps provenance and prompt-format tooling usable in non-Metal environments;
the Apple-Silicon model is loaded only when the v0 chat runtime is requested.
"""

from cyber_agent.inference.prompt import ChatHistory, DEFAULT_SYSTEM_PROMPT

__all__ = ["ChatHistory", "DEFAULT_SYSTEM_PROMPT"]
