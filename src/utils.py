# src/utils.py
import os
from datetime import datetime

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def timestamped_filename(prefix: str, ext: str = ".pkl"):
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}{ext}"
