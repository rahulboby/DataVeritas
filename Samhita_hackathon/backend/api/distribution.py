from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import dataset_service as ds
from services import analytics_service as analytics

router = APIRouter(prefix="/api", tags=["Distribution"])


class DistributionRequest(BaseModel):
    dataset_id: str


@router.post("/distribution")
async def compute_distribution(req: DistributionRequest):
    """Column type distribution and statistics."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        return analytics.compute_distribution_analysis(df)
    except Exception as e:
        raise HTTPException(500, f"Error analyzing distribution: {str(e)}")


@router.post("/distribution/profile")
async def compute_profile(req: DistributionRequest):
    """Full dataset schema profiling."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        from core.profiling.schema_detector import detect_schema
        return detect_schema(df)
    except Exception as e:
        raise HTTPException(500, f"Error profiling schema: {str(e)}")
