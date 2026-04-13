"""Celery application configuration for Constellation workers."""

import logging

from celery import Celery

from constellation.config import get_settings

# Silence chatty third-party loggers — httpx/httpcore emit one INFO line
# per HTTP call, which is overwhelming for embedding workloads.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

settings = get_settings()

celery_app = Celery(
    "constellation",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)

celery_app.autodiscover_tasks(["constellation.worker"])
