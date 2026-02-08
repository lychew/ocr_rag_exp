"""Configuration loader — merges YAML configs into a single dict."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def load_config(
    ocr_name: str | None = None,
    chunking_name: str | None = None,
) -> dict:
    """Load default config and optionally overlay OCR / chunking overrides."""
    cfg = _load_yaml(_CONFIG_DIR / "default.yaml")

    if ocr_name:
        override = _CONFIG_DIR / f"ocr_{ocr_name}.yaml"
        if override.exists():
            cfg = _deep_merge(cfg, _load_yaml(override))
        cfg.setdefault("ocr", {})["name"] = ocr_name

    if chunking_name:
        override = _CONFIG_DIR / f"chunking_{chunking_name}.yaml"
        if override.exists():
            cfg = _deep_merge(cfg, _load_yaml(override))
        cfg.setdefault("chunking", {})["name"] = chunking_name

    # Resolve relative paths against project root
    for key in ("pdf_path", "data_dir"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = str(_PROJECT_ROOT / cfg[key])

    return cfg


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
