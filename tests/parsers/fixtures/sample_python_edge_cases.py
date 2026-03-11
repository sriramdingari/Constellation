"""Edge cases for Python parser testing."""

from enum import Enum
from celery import shared_task
from rest_framework.decorators import api_view


class Product(models.Model):
    """A Django model."""

    name = models.CharField(max_length=100)
    price = models.DecimalField()


class UserSchema(BaseModel):
    """A Pydantic model."""

    username: str
    email: str


@shared_task
def send_email(to: str, subject: str) -> bool:
    """A Celery task."""
    return True


@api_view(["GET"])
def list_products(request) -> dict:
    """An API endpoint."""
    return {"products": []}


class OuterClass:
    """A class with a nested inner class."""

    class InnerClass:
        """An inner class."""

        def inner_method(self) -> None:
            pass

    def outer_method(self) -> None:
        pass


class Color(Enum):
    """An enum class."""

    RED = 1
    GREEN = 2
    BLUE = 3


class Calculator:
    """A class with field extraction from __init__."""

    def __init__(self, precision: int):
        self.precision = precision
        self.history = []
        self._cache = {}

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = self._compute(a, b)
        self._store_result(result)
        return result

    def _compute(self, a: int, b: int) -> int:
        return a + b

    def _store_result(self, result: int) -> None:
        self.history.append(result)
