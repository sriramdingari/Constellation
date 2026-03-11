"""Celery application configuration for Constellation workers."""

from celery import Celery

from constellation.config import get_settings

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
