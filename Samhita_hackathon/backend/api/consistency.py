from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services import dataset_service as ds
from services import analytics_service as analytics

router = APIRouter(prefix="/api", tags=["Consistency"])


class ConsistencyRequest(BaseModel):
    dataset_id: str
    rules: Optional[List[dict]] = None


class RuleDiscoveryRequest(BaseModel):
    dataset_id: str


@router.post("/consistency")
async def compute_consistency(req: ConsistencyRequest):
    """Consistency analysis with optional custom rules."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        return analytics.compute_consistency_analysis(df, rules=req.rules)
    except Exception as e:
        raise HTTPException(500, f"Error analyzing consistency: {str(e)}")


@router.post("/consistency/discover-rules")
async def discover_rules(req: RuleDiscoveryRequest):
    """Automatically discover data quality rules."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        from core.consistency.rule_discovery import discover_rules
        rules = discover_rules(df)
        return {
            "discovered_rules": rules,
            "total_rules": len(rules),
        }
    except Exception as e:
        raise HTTPException(500, f"Error discovering rules: {str(e)}")
