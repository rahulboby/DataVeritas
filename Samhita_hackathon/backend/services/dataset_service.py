import pandas as pd
import uuid
import io
from typing import Optional, Dict


# In-memory dataset store
_datasets: Dict[str, pd.DataFrame] = {}
_metadata: Dict[str, dict] = {}


def store_dataset(df: pd.DataFrame, filename: str = "uploaded") -> str:
    """
    Store a DataFrame in memory and return a dataset_id.
    """
    dataset_id = str(uuid.uuid4())[:8]
    _datasets[dataset_id] = df
    _metadata[dataset_id] = {
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    return dataset_id


def get_dataset(dataset_id: str) -> Optional[pd.DataFrame]:
    """Retrieve a stored DataFrame by ID."""
    return _datasets.get(dataset_id)


def get_metadata(dataset_id: str) -> Optional[dict]:
    """Retrieve dataset metadata."""
    return _metadata.get(dataset_id)


def delete_dataset(dataset_id: str) -> bool:
    """Remove a dataset from memory."""
    if dataset_id in _datasets:
        del _datasets[dataset_id]
        del _metadata[dataset_id]
        return True
    return False


def list_datasets() -> list:
    """List all stored datasets."""
    return [
        {"dataset_id": did, **meta}
        for did, meta in _metadata.items()
    ]


def parse_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded file bytes into a DataFrame."""
    name_lower = filename.lower()
    if name_lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_content))
    elif name_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_content))
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def get_preview(dataset_id: str, limit: int = 100) -> Optional[dict]:
    """Get a preview of the dataset."""
    df = get_dataset(dataset_id)
    if df is None:
        return None
    
    preview_df = df.head(limit)
    
    # Convert to JSON-safe format
    rows = []
    for _, row in preview_df.iterrows():
        row_dict = {}
        for col in preview_df.columns:
            val = row[col]
            if pd.isnull(val):
                row_dict[col] = None
            elif hasattr(val, 'isoformat'):
                row_dict[col] = val.isoformat()
            else:
                row_dict[col] = str(val) if not isinstance(val, (int, float, bool)) else val
        rows.append(row_dict)
    
    return {
        "rows": rows,
        "total": len(df),
        "showing": len(preview_df),
        "columns": df.columns.tolist(),
    }
