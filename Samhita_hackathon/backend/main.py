from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.upload import router as upload_router
from api.score import router as score_router
from api.nulls import router as nulls_router
from api.duplicates import router as duplicates_router
from api.outliers import router as outliers_router
from api.consistency import router as consistency_router
from api.distribution import router as distribution_router
from api.insights import router as insights_router


app = FastAPI(
    title="DataVeritas API",
    description="Enterprise Data Quality & Trust Analytics Platform",
    version="2.0.0",
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload_router)
app.include_router(score_router)
app.include_router(nulls_router)
app.include_router(duplicates_router)
app.include_router(outliers_router)
app.include_router(consistency_router)
app.include_router(distribution_router)
app.include_router(insights_router)


@app.get("/")
async def root():
    return {
        "name": "DataVeritas API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
