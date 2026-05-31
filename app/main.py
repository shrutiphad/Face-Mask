import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.embedder import FaceEmbedder
from app.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting — model: %s  threshold: %.2f",
                settings.insightface_model_pack, settings.match_threshold)
    app.state.embedder = FaceEmbedder()
    app.state.vector_store = VectorStore()
    count = app.state.vector_store.count()
    logger.info("Ready — %d identities enrolled.", count)
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(title="Face Match + Vector Search", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-t0)*1000:.1f}"
        return response

    @app.get("/health")
    async def health(request: Request):
        try:
            count = request.app.state.vector_store.count()
            return {"ok": True, "enrolled_count": count}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/enroll")
    async def enroll(
        request: Request,
        id: str,
        image: UploadFile = File(...),
    ):
        logger.info("ENROLL id=%s file=%s", id, image.filename)

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="Empty file.")

        try:
            embedding = request.app.state.embedder.embed(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        request.app.state.vector_store.upsert(id, embedding)
        count = request.app.state.vector_store.count()
        logger.info("ENROLL OK id=%s enrolled=%d", id, count)

        return {
            "id": id,
            "stored": True,
            "enrolled_count": count,
        }

    @app.post("/search")
    async def search(
        request: Request,
        image: UploadFile = File(...),
    ):
        settings = get_settings()
        logger.info("SEARCH file=%s", image.filename)

        count = request.app.state.vector_store.count()
        if count == 0:
            raise HTTPException(
                status_code=422,
                detail="No faces enrolled yet. Call POST /enroll first.",
            )

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="Empty file.")

        try:
            embedding = request.app.state.embedder.embed(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        t0 = time.perf_counter()
        hits = request.app.state.vector_store.search(
            query_embedding=embedding,
            top_k=settings.top_k,
            ef_search=settings.hnsw_ef_search,
        )
        search_ms = (time.perf_counter() - t0) * 1000

        match_id = None
        cosine   = 0.0
        is_match = False

        if hits:
            match_id = hits[0]["identity_id"]
            cosine   = hits[0]["score"]
            is_match = cosine >= settings.match_threshold

        logger.info("SEARCH match_id=%s cosine=%.4f is_match=%s search_ms=%.1f",
                    match_id, cosine, is_match, search_ms)

        return {
            "match_id":  match_id,
            "cosine":    round(cosine, 6),
            "is_match":  is_match,
            "threshold": settings.match_threshold,
            "search_latency_ms": round(search_ms, 2),
            "enrolled_count": count,
        }

    return app


app = create_app()