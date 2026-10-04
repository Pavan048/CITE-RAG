"""FastAPI application entrypoint. Owns the lifespan of the in-process ingestion worker pool
(app/ingestion/pipeline.py) — per the "use FastAPI only" instruction, there is no separate
Celery/RQ worker process to start; the workers are asyncio tasks living inside this same process.

Also serves the built frontend (frontend/dist/, built via `npm run build` from frontend/src/). The
PRD's scope (Section 4) only specifies the API — this exists purely so the running system is
usable from a browser, not as a PRD requirement. The static mount is registered *after* the API
routers below so an incoming request checks `/v1/...` routes first — Starlette matches routes in
registration order, and a root-mounted StaticFiles would otherwise shadow everything.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import documents, jobs, query
from app.ingestion.pipeline import recover_stranded_jobs, start_workers, stop_workers
from app.services.mongo_client import mongo_service
from app.services.qdrant_client import qdrant_service

_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    qdrant_service.ensure_collections()
    mongo_service.ensure_indexes()
    # Before starting fresh workers: fail out any job stranded by the *previous* process's exit
    # (see recover_stranded_jobs()'s docstring for why that's a clean failure, not a resume).
    recover_stranded_jobs()
    start_workers()
    yield
    await stop_workers()


app = FastAPI(title="PDF Knowledge Base RAG System", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(jobs.router)
app.include_router(query.router)

# `html=True` serves frontend/dist/index.html for `/` (and for any other unmatched path, so
# client-side routing — if this ever grows beyond a single page — wouldn't 404 on refresh).
app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
