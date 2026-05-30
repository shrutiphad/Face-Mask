 
import logging
import time
from contextlib import asynccontextmanager
 
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
 
from app.config import get_settings
from app.embedder import FaceEmbedder
from app.schemas import EnrollResponse, ErrorResponse, SearchMatch, SearchResponse
from app.vector_store import VectorStore
 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
 

 
@asynccontextmanager
async def lifespan(app: FastAPI):
   
    settings = get_settings()
    logger.info("=== Face Match Service starting up ===")
    logger.info("Qdrant URL : %s", settings.qdrant_url)
    logger.info("Model pack : %s", settings.insightface_model_pack)
    logger.info("Threshold  : %.2f", settings.match_threshold)
 
    # Heavy initialisation — happens ONCE per process, not per request
    app.state.embedder = FaceEmbedder()
    app.state.vector_store = VectorStore()
 
    count = app.state.vector_store.count()
    logger.info("Startup complete — %d identities enrolled.", count)
 
    yield  # ← application runs here
 
    logger.info("=== Face Match Service shutting down ===")

 
 
#  Application factory 
 
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Face Match Service",
        description=(
            "ArcFace + Qdrant HNSW face identification API.\n\n"
            "**Enroll** a face image under an identity ID, then **search** "
            "with a query image to get the best cosine-similarity match."
        ),
        version="1.0.0",
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
 
    # CORS — wide-open for the assignment; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
 
    #  Request timing middleware 
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """
        Adds X-Process-Time-Ms header to every response.
        Useful when curl-ing the API — you can see total latency at a glance.
        """
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response
 
    #  Endpoints 
 
    @app.get("/health", tags=["ops"])
    async def health(request: Request):
        """
        Liveness + readiness probe.
        Returns 200 with enrolled count when the service is fully operational.
        Returns 503 if Qdrant is unreachable.
        """
        try:
            count = request.app.state.vector_store.count()
            return {"status": "ok", "enrolled_count": count}
        except Exception as exc:
            logger.exception("Health check failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Qdrant unreachable: {exc}",
            )
 
    @app.get("/identities", tags=["ops"])
    async def list_count(request: Request):
        """Return how many faces are currently enrolled."""
        count = request.app.state.vector_store.count()
        return {"enrolled_count": count}
 
    
    @app.post(
        "/enroll",
        response_model=EnrollResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["face-match"],
        summary="Enroll a face image under an identity ID",
    )
    async def enroll(
        request: Request,
        image: UploadFile = File(
            ...,
            description="Face image (JPEG / PNG / WebP). Must contain exactly one face.",
        ),
        identity_id: str = Form(
            ...,
            description="Unique string label for this person, e.g. 'person_01'.",
            min_length=1,
            max_length=128,
        ),
    ) -> EnrollResponse:
        """
        Detect, align, and embed the uploaded face, then store the 512-d
        ArcFace vector in Qdrant under `identity_id`.
 
        Re-enrolling the same `identity_id` with a new photo **replaces** the
        stored embedding (upsert semantics) — no duplicates are created.
        """
        logger.info("ENROLL request — identity='%s' file='%s'", identity_id, image.filename)
        t_start = time.perf_counter()
 
        #  Read raw bytes from the uploaded file 
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty file — please upload a non-empty image.",
            )
 
        #  Embed (detect + align + ArcFace) 
        try:
            embedding = request.app.state.embedder.embed(image_bytes)
        except ValueError as exc:
            # ValueError = no face detected or bad image — client error
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
 
        embed_ms = (time.perf_counter() - t_start) * 1000
 
        #  Store in Qdrant 
        t_store = time.perf_counter()
        request.app.state.vector_store.upsert(identity_id, embedding)
        store_ms = (time.perf_counter() - t_store) * 1000
 
        count = request.app.state.vector_store.count()
        total_ms = (time.perf_counter() - t_start) * 1000
 
        logger.info(
            "ENROLL OK — identity='%s' embed_ms=%.1f store_ms=%.1f total_ms=%.1f enrolled=%d",
            identity_id, embed_ms, store_ms, total_ms, count,
        )
 
        return EnrollResponse(
            identity_id=identity_id,
            enrolled_count=count,
            message=f"enrolled successfully in {total_ms:.0f} ms",
        )
 
    
    @app.post(
        "/search",
        response_model=SearchResponse,
        status_code=status.HTTP_200_OK,
        tags=["face-match"],
        summary="Search for the closest enrolled face",
    )
    async def search(
        request: Request,
        image: UploadFile = File(
            ...,
            description="Query face image (JPEG / PNG / WebP).",
        ),
    ) -> SearchResponse:
        
        settings = get_settings()
        logger.info("SEARCH request — file='%s'", image.filename)
        t_start = time.perf_counter()
 
        #  Sanity: collection must have at least one vector 
        count = request.app.state.vector_store.count()
        if count == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No faces enrolled yet. Call POST /enroll first.",
            )
 
        #  Read + embed query image 
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Empty file.",
            )
 
        try:
            embedding = request.app.state.embedder.embed(image_bytes)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
 
        embed_ms = (time.perf_counter() - t_start) * 1000
 
        # ── ANN search (this is the Qdrant HNSW call — NOT a loop) ────────────
        t_search = time.perf_counter()
        hits = request.app.state.vector_store.search(
            query_embedding=embedding,
            top_k=settings.top_k,
            ef_search=settings.hnsw_ef_search,
        )
        search_ms = (time.perf_counter() - t_search) * 1000
 
        total_ms = (time.perf_counter() - t_start) * 1000
 
        #  Build response
        top_match: SearchMatch | None = None
        is_match = False
        if hits:
            top = hits[0]
            top_match = SearchMatch(
                identity_id=top["identity_id"],
                score=top["score"],
            )
            is_match = top["score"] >= settings.match_threshold
 
        logger.info(
            "SEARCH OK — match=%s score=%s is_match=%s "
            "embed_ms=%.1f search_ms=%.1f total_ms=%.1f",
            top_match.identity_id if top_match else "none",
            f"{top_match.score:.4f}" if top_match else "n/a",
            is_match,
            embed_ms,
            search_ms,
            total_ms,
        )
 
        return SearchResponse(
            query_file=image.filename or "unknown",
            top_match=top_match,
            is_match=is_match,
            threshold_used=settings.match_threshold,
            latency_ms=round(search_ms, 2),
            enrolled_count=count,
        )
 
    return app
 
 
app = create_app()
 