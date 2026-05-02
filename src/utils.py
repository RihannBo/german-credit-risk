from pathlib import Path
from typing import Any

import joblib


def save_object(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)
