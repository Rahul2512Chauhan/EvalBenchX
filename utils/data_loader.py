import os
import pandas as pd
from typing import Optional
from config import (
    DEFAULT_DATA_PATH,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    SUPPORTED_FORMATS
)

def load_dataset(file_path: Optional[str] = None, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load and validate a dataset file (CSV or JSON), with optional row limit.
    
    Parameters:
        file_path (str): Optional path to the dataset file. If not provided, uses default.
        max_rows (int): Optional limit on number of rows to load.
        
    Returns:
        pd.DataFrame: Cleaned dataset with required and optional columns.
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format unsupported or required columns missing
    """

    # Use default path if not provided
    path = file_path or DEFAULT_DATA_PATH

    # File existence check
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # Format check
    ext = os.path.splitext(path)[1]
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_FORMATS}")
    
    # Load data based on format
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format after extension check: {ext}")

    # Limit rows if specified
    if max_rows:
        df = df.iloc[:max_rows]

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Fill optional columns
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Reorder columns
    final_cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    df = df[final_cols]

    return df
