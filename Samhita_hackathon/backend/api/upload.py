from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from services import dataset_service as ds

router = APIRouter(prefix="/api", tags=["Dataset"])


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or Excel file for analysis."""
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    allowed_extensions = ('.csv', '.xlsx', '.xls')
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(400, f"Unsupported file type. Allowed: {allowed_extensions}")
    
    try:
        content = await file.read()
        df = ds.parse_file(content, file.filename)
        dataset_id = ds.store_dataset(df, file.filename)
        meta = ds.get_metadata(dataset_id)
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            **meta
        }
    except Exception as e:
        raise HTTPException(400, f"Error processing file: {str(e)}")


@router.post("/generate")
async def generate_dataset(n_rows: int = 20000):
    """Generate synthetic test data."""
    if n_rows < 100 or n_rows > 200000:
        raise HTTPException(400, "Row count must be between 100 and 200,000")
    
    from core.data.generator import get_data
    df = get_data(n_rows)
    dataset_id = ds.store_dataset(df, f"generated_{n_rows}_rows")
    meta = ds.get_metadata(dataset_id)
    
    return {
        "dataset_id": dataset_id,
        **meta
    }


@router.get("/preview")
async def preview_dataset(dataset_id: str, limit: int = Query(100, ge=1, le=10000)):
    """Get a preview of the uploaded dataset."""
    result = ds.get_preview(dataset_id, limit)
    if result is None:
        raise HTTPException(404, "Dataset not found")
    return result


@router.get("/datasets")
async def list_datasets():
    """List all stored datasets."""
    return ds.list_datasets()


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset from memory."""
    if ds.delete_dataset(dataset_id):
        return {"message": "Dataset deleted"}
    raise HTTPException(404, "Dataset not found")
