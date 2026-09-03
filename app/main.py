import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.config import settings
from app.services.ai_pipeline import ai_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

from fastapi.responses import RedirectResponse

# CORS middleware
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # Allow all origins for development
        allow_credentials=False, # Must be False if allow_origins is ["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing AI Pipeline...")
    ai_pipeline.load_model()
    logger.info("Application startup complete.")
