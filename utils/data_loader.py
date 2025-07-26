import os
import pandas as pd
from typing import Optional
from config import (
    DEFAULT_DATA_PATH,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    SUPPORTED_FORMATS
)

def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and validate a dataset file (CSV or JSON).
    
    Parameters:
        file_path (str): Optional path to the dataset file. If not provided, uses default.
        
    Returns:
        pd.DataFrame: Cleaned dataset with required and optional columns.
    
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format unsupported or required columns missing
    """

    #use default if no file path given
    path = file_path or DEFAULT_DATA_PATH

    #check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    #check file format
    ext = os.path.splitext(path)[1]
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_FORMATS}")
    
    # Load CSV or JSON
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path, lines=True) if path.endswith(".jsonl") else pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format after extension check: {ext}")


    #check for reuired columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    #fill missing optional columns with empty strings
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    #reorder coulmns
    final_cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    df = df[final_cols]

    return df 
    