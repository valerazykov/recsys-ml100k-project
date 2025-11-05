from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import yaml


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across random, numpy and torch.

    Note: full determinism in PyTorch may require disabling benchmark and setting
    deterministic flags. This can slow down training.
    """
    try:
        import torch
    except Exception:
        torch = None

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set python and numpy seed to {seed}")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Try to force deterministic behavior (may slow down)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Configured torch.backends.cudnn for deterministic behavior (may be slower).")
        except Exception:
            logger.warning("Could not set cudnn deterministic flags.")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist (no-op if exists)."""
    if path is None or path == "":
        return
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str, indent: int = 2, ensure_ascii: bool = False) -> None:
    """Save Python object as JSON to disk (creates parent dir if needed)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)
    logger.debug(f"Saved JSON to {path}")


def load_json(path: str) -> Any:
    """Load JSON file from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON from {path}")
    return data


def save_yaml(obj: Any, path: str) -> None:
    """Save object as YAML to disk (creates parent dir if needed)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    logger.debug(f"Saved YAML to {path}")


def load_yaml(path: str) -> Any:
    """Load YAML file from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug(f"Loaded YAML from {path}")
    return data


def setup_basic_logger(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup root logger: StreamHandler (console) + optional FileHandler.
    Level is a string like "INFO" or "DEBUG".
    """
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    level_value = getattr(logging, level.upper(), logging.INFO)
    # configure root logger only if not configured
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level_value)
        ch = logging.StreamHandler()
        ch.setLevel(level_value)
        ch.setFormatter(logging.Formatter(fmt))
        root.addHandler(ch)
        if log_file:
            ensure_dir(os.path.dirname(log_file) or ".")
            fh = logging.FileHandler(log_file)
            fh.setLevel(level_value)
            fh.setFormatter(logging.Formatter(fmt))
            root.addHandler(fh)
    else:
        # if already configured, just set level and optionally add file handler
        root.setLevel(level_value)
        if log_file:
            # add file handler if not present
            if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in root.handlers):
                ensure_dir(os.path.dirname(log_file) or ".")
                fh = logging.FileHandler(log_file)
                fh.setLevel(level_value)
                fh.setFormatter(logging.Formatter(fmt))
                root.addHandler(fh)
    logger.debug(f"Logger configured. level={level}, log_file={log_file}")


def dict_deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dictionary 'base' with 'updates' and return updated dict.
    Similar to dict.update but handles nested dicts.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = dict_deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def safe_cast(obj: Any, dtype: type, default: Optional[Any] = None) -> Any:
    """
    Try to cast obj to dtype, return default if casting fails.
    """
    try:
        return dtype(obj)
    except Exception:
        return default
