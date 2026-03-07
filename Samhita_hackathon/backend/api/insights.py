from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services import dataset_service as ds

router = APIRouter(prefix="/api", tags=["Insights"])


class InsightRequest(BaseModel):
    dataset_id: str
    rules: Optional[List[dict]] = None


@router.post("/insights")
async def generate_insights(req: InsightRequest):
    """Generate AI-driven data quality insights."""
    df = ds.get_dataset(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset not found")
    
    try:
        from core.insights.insight_engine import generate_insights
        insights = generate_insights(df, rules=req.rules)
        return {
            "insights": insights,
            "total": len(insights),
            "by_severity": {
                "critical": len([i for i in insights if i["severity"] == "critical"]),
                "warning": len([i for i in insights if i["severity"] == "warning"]),
                "info": len([i for i in insights if i["severity"] == "info"]),
                "good": len([i for i in insights if i["severity"] == "good"]),
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Error generating insights: {str(e)}")
