"""FastAPI application entry point for Constellation."""

from fastapi import FastAPI

from constellation.api.routes import router

app = FastAPI(title="Constellation", version="0.1.0")
app.include_router(router)
