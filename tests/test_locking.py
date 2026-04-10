"""Tests for Redis-backed lock heartbeat helpers."""

import time
from unittest.mock import MagicMock

from constellation.locking import start_claim_heartbeat, start_lock_heartbeat


class TestClaimHeartbeat:
    def test_refreshes_claim_until_stopped(self):
        redis_client = MagicMock()
        redis_client.get.return_value = "token-123"
        redis_client.expire = MagicMock()

        stop = start_claim_heartbeat(
            redis_client,
            "constellation:lock:repo",
            "token-123",
            timeout=30,
            interval=0.01,
        )
        try:
            time.sleep(0.03)
        finally:
            stop()

        assert redis_client.expire.call_count >= 1

    def test_stops_when_callback_returns_false(self):
        redis_client = MagicMock()
        redis_client.get.return_value = "token-123"
        redis_client.expire = MagicMock()
        keep_running = iter([True, False])

        stop = start_claim_heartbeat(
            redis_client,
            "constellation:lock:repo",
            "token-123",
            timeout=30,
            interval=0.01,
            should_continue=lambda: next(keep_running, False),
        )
        try:
            time.sleep(0.04)
        finally:
            stop()

        assert redis_client.expire.call_count == 1

    def test_respects_max_lifetime(self):
        redis_client = MagicMock()
        redis_client.get.return_value = "token-123"
        redis_client.expire = MagicMock()

        stop = start_claim_heartbeat(
            redis_client,
            "constellation:lock:repo",
            "token-123",
            timeout=30,
            interval=0.01,
            max_lifetime=0.005,
        )
        try:
            time.sleep(0.03)
        finally:
            stop()

        redis_client.expire.assert_not_called()


class TestLockHeartbeat:
    def test_refreshes_direct_worker_lock_until_stopped(self):
        lock = MagicMock()
        lock.extend = MagicMock()

        stop = start_lock_heartbeat(lock, timeout=30, interval=0.01)
        try:
            time.sleep(0.03)
        finally:
            stop()

        assert lock.extend.call_count >= 1
