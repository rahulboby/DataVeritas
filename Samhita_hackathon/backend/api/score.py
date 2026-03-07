from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from services import dataset_service as ds
from services import analytics_service as analytics

router = APIRouter(prefix="/api", tags=["Scoring"])


class ScoreRequest(BaseModel):
    dataset_id: str


class FieldScoreRequest(BaseModel):
    dataset_id: str
    column: str


@router.post("/score")
async def compute_score(req: ScoreRequest):
    """Compute overall DQ score and all sub-scores."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        return analytics.compute_scores(df)
    except Exception as e:
        raise HTTPException(500, f"Error computing score: {str(e)}")


@router.post("/score/field")
async def compute_field_score(req: FieldScoreRequest):
    """Compute per-field DQ score."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        return analytics.compute_field_score(df, req.column)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error computing field score: {str(e)}")


@router.post("/score/dynamic")
async def compute_dynamic_score(req: ScoreRequest):
    """Compute dynamic entropy-based trust score."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        from core.score.dynamic_weights import compute_dynamic_weights
        return compute_dynamic_weights(df)
    except Exception as e:
        raise HTTPException(500, f"Error computing dynamic score: {str(e)}")


@router.post("/score/predict")
async def predict_quality(req: ScoreRequest):
    """AI-predicted quality score using meta-features."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        from core.ml.quality_predictor import predict_quality
        return predict_quality(df)
    except Exception as e:
        raise HTTPException(500, f"Error predicting quality: {str(e)}")
