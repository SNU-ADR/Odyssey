"""Stable track-token -> int64 id (stdlib-only: shared by target builder, live PDM GT, loss)."""
import hashlib


def token_to_id(token: str) -> int:
    if not token:
        return 0
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "little", signed=True)
