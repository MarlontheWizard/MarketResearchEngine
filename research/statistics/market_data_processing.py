#Author: Marlon Dominguez
#Date  : 08/13/2026

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ColumnConfig

#Join context and confirmation data together
def load_and_join(context_path: str | Path, lifecycle_path: str | Path | None, columns: ColumnConfig) -> pd.DataFrame:
    
    context = _read_table(context_path)
    
    if lifecycle_path is None:
        
        return prepare_frame(context, columns)
    
    lifecycle = _read_table(lifecycle_path)
    timestamp = columns.timestamp
    
    if timestamp not in context or timestamp not in lifecycle:
        
        raise ValueError(f"Both inputs must contain {timestamp!r}")
    
    lifecycle_only = [name for name in lifecycle.columns if name == timestamp or name not in context.columns]
    merged = context.merge(lifecycle[lifecycle_only], on=timestamp, how="inner", validate="one_to_one")
    
    if len(merged) != len(lifecycle):
        
        raise ValueError("[ERROR] CONTEXT AND LIFECYCLE TIMESTAMPS DO NOT ALIGN 1:1")
    
    return prepare_frame(merged, columns)


def prepare_frame(frame: pd.DataFrame, columns: ColumnConfig) -> pd.DataFrame:
    
    required = [columns.timestamp, columns.open, columns.high, columns.low, columns.close,
        	columns.confirmed_now, columns.first_tradable_now, columns.active_live,
        	columns.invalidated_now, columns.confirmed_upper, columns.confirmed_lower
               ]
    
    missing = [name for name in required if name not in frame]
    
    if missing:
        
        raise ValueError(f"[ERROR] MISSING REQUIRED COLUMN(S): {missing}")
    
    result = frame.copy()
   
    result[columns.timestamp] = pd.to_datetime(result[columns.timestamp], errors="raise", utc=True)
    
    result = result.sort_values(columns.timestamp, kind="stable").reset_index(drop=True)
    
    if result[columns.timestamp].duplicated().any():
        
        raise ValueError("[ERROR] DUPLICATE TIMESTAMPS NOT ALLOWED")
        
    for name in (columns.confirmed_now, columns.first_tradable_now, columns.active_live, columns.invalidated_now):
        
        result[name] = result[name].fillna(False).astype(bool)
        
    if columns.atr not in result:
        
        result[columns.atr] = pd.NA
    
    
    return result



def read_table(path: str | Path) -> pd.DataFrame:
    
    path = Path(path)
    
    suffix = path.suffix.lower()
    
    #most likely will have mixed file types
    if suffix in {".parquet", ".pq"}:
        
        return pd.read_parquet(path)
    
    if suffix == ".csv":
        
        return pd.read_csv(path)
    
    raise ValueError(f"[ERROR] UNSUPPORTED INPUT FORMAT: {path.suffix}")
