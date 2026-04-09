"""Redis-backed indexing claim helpers shared by the API and worker."""

from __future__ import annotations

from math import ceil
from threading import Event, Thread
from time import monotonic

INDEX_LOCK_TTL_SECONDS = 3600
HEARTBEAT_INTERVAL_SECONDS = INDEX_LOCK_TTL_SECONDS / 3


def _decode_lock_value(value: bytes | str | None) -> str | None:
    """Normalize Redis values so token comparisons work across client modes."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def acquire_dispatch_claim(
    redis_client,
    lock_key: str,
    token: str,
    *,
    timeout: int = INDEX_LOCK_TTL_SECONDS,
) -> bool:
    """Atomically reserve a repository before queueing an indexing task."""
    return bool(redis_client.set(lock_key, token, nx=True, ex=timeout))


def acquire_worker_claim(
    redis_client,
    lock_key: str,
    token: str,
    *,
    timeout: int = INDEX_LOCK_TTL_SECONDS,
) -> bool:
    """Claim or resume ownership of a repository lock inside the worker."""
    current_token = _decode_lock_value(redis_client.get(lock_key))
    if current_token is None:
        return bool(redis_client.set(lock_key, token, nx=True, ex=timeout))
    if current_token != token:
        return False

    redis_client.expire(lock_key, timeout)
    return True


def refresh_claim(
    redis_client,
    lock_key: str,
    token: str,
    *,
    timeout: int = INDEX_LOCK_TTL_SECONDS,
) -> bool:
    """Refresh a claim TTL only if the caller still owns it."""
    current_token = _decode_lock_value(redis_client.get(lock_key))
    if current_token != token:
        return False

    redis_client.expire(lock_key, timeout)
    return True


def start_claim_heartbeat(
    redis_client,
    lock_key: str,
    token: str,
    *,
    timeout: int = INDEX_LOCK_TTL_SECONDS,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
    should_continue=None,
    max_lifetime: float | None = None,
):
    """Start a background heartbeat that keeps a claim alive while work runs."""
    stop_event = Event()
    started_at = monotonic()

    def _heartbeat() -> None:
        while not stop_event.wait(interval):
            try:
                if should_continue is not None and not should_continue():
                    return
                refresh_timeout = timeout
                if max_lifetime is not None:
                    remaining = max_lifetime - (monotonic() - started_at)
                    if remaining <= 0:
                        return
                    refresh_timeout = min(timeout, max(1, ceil(remaining)))
                if not refresh_claim(
                    redis_client,
                    lock_key,
                    token,
                    timeout=refresh_timeout,
                ):
                    return
            except Exception:
                return

    thread = Thread(target=_heartbeat, daemon=True)
    thread.start()

    def _stop() -> None:
        stop_event.set()
        thread.join(timeout=interval)

    return _stop


def start_lock_heartbeat(
    lock,
    *,
    timeout: int = INDEX_LOCK_TTL_SECONDS,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
):
    """Start a background heartbeat for direct worker locks."""
    stop_event = Event()

    def _heartbeat() -> None:
        while not stop_event.wait(interval):
            try:
                lock.extend(timeout, replace_ttl=True)
            except Exception:
                return

    thread = Thread(target=_heartbeat, daemon=True)
    thread.start()

    def _stop() -> None:
        stop_event.set()
        thread.join(timeout=interval)

    return _stop


def release_claim(redis_client, lock_key: str, token: str) -> None:
    """Release a repository claim only if the caller still owns it."""
    current_token = _decode_lock_value(redis_client.get(lock_key))
    if current_token == token:
        redis_client.delete(lock_key)
