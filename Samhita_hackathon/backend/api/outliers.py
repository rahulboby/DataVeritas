from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import dataset_service as ds
from services import analytics_service as analytics

router = APIRouter(prefix="/api", tags=["Outliers"])


class OutlierRequest(BaseModel):
    dataset_id: str


@router.post("/outliers")
async def compute_outliers(req: OutlierRequest):
    """Comprehensive outlier analysis."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        return analytics.compute_outlier_analysis(df)
    except Exception as e:
        raise HTTPException(500, f"Error analyzing outliers: {str(e)}")
