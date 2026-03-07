from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services import dataset_service as ds
from services import analytics_service as analytics

router = APIRouter(prefix="/api", tags=["Duplicates"])


class DuplicateRequest(BaseModel):
    dataset_id: str


class EntityGraphRequest(BaseModel):
    dataset_id: str
    key_columns: Optional[List[str]] = None
    similarity_threshold: int = 80


@router.post("/duplicates")
async def compute_duplicates(req: DuplicateRequest):
    """Comprehensive duplicate analysis."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        return analytics.compute_duplicate_analysis(df)
    except Exception as e:
        raise HTTPException(500, f"Error analyzing duplicates: {str(e)}")


@router.post("/duplicates/entity-graph")
async def compute_entity_graph(req: EntityGraphRequest):
    """Graph-based entity resolution."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        from core.duplicates.entity_graph import build_entity_graph
        return build_entity_graph(
            df, 
            key_columns=req.key_columns,
            similarity_threshold=req.similarity_threshold
        )
    except Exception as e:
        raise HTTPException(500, f"Error building entity graph: {str(e)}")
