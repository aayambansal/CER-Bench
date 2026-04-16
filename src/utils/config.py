"""Configuration loader for SynthSearch."""

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML config file from configs/ directory.

    Args:
        name: Config name without extension (e.g., 'corpus', 'retrieval').

    Returns:
        Parsed config dictionary.
    """
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_data_dir(subdir: str = "") -> Path:
    """Get a data directory path, creating it if needed."""
    path = PROJECT_ROOT / "data" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_dir(subdir: str = "") -> Path:
    """Get a results directory path, creating it if needed."""
    path = PROJECT_ROOT / "results" / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_env_or_raise(var: str) -> str:
    """Get an environment variable or raise with helpful message."""
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Environment variable {var} not set. "
            f"Set it with: export {var}='your-key-here'"
        )
    return val
