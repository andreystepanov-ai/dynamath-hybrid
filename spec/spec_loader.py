# spec/spec_loader.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).parent

def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
        with path.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        import toml  # pip install toml
        with path.open("r", encoding="utf-8") as f:
            return toml.load(f)

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_project_config(name: str = "hybrid.toml") -> Dict[str, Any]:
    p = BASE_DIR / name
    return _load_toml(p) if p.exists() else {}

def get_operator_registry(name: str = "operators.json") -> Dict[str, Any]:
    p = BASE_DIR / name
    return _load_json(p) if p.exists() else {"tensor": [], "quantum": []}
