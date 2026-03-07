from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services import dataset_service as ds
from services import analytics_service as analytics

router = APIRouter(prefix="/api", tags=["Nulls"])


class NullRequest(BaseModel):
    dataset_id: str


@router.post("/nulls")
async def compute_nulls(req: NullRequest):
    """Comprehensive null/completeness analysis."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        return analytics.compute_null_analysis(df)
    except Exception as e:
        raise HTTPException(500, f"Error analyzing nulls: {str(e)}")
