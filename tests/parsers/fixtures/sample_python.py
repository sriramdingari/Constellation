"""Sample Python module for testing the parser."""

import os
from typing import Optional, List
from dataclasses import dataclass


class UserService:
    """Service for managing users."""

    def __init__(self, db_connection: str, timeout: int = 30):
        """Initialize the user service."""
        self.db_connection = db_connection
        self.timeout = timeout

    def get_user(self, user_id: int) -> Optional[dict]:
        """Fetch a single user by ID."""
        result = self.query_db(user_id)
        return result

    async def list_users(self, limit: int = 100) -> List[dict]:
        """List all users with an optional limit."""
        return []


def process_data(items: List[str], verbose: bool = False) -> int:
    """Process a list of data items.

    Args:
        items: The items to process.
        verbose: Whether to log details.

    Returns:
        The number of items processed.
    """
    count = len(items)
    return count


class TestUserService:
    """Tests for UserService."""

    def test_get_user(self):
        """Test that get_user returns a user dict."""
        service = UserService("sqlite://")
        result = service.get_user(1)
        assert result is not None
