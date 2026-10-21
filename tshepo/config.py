"""Paths and shared settings.

Everything the project writes goes under the repository root:
    data/    generated or hand-labelled message files (never commit real data)
    models/  trained artefacts (tfidf.joblib, hybrid.joblib, transformer/, score models)

Configuration precedence (highest wins):
  1. TSHEPO_* environment variables
  2. tshepo.yaml / tshepo.json config file (looked up in cwd, then repo root)
  3. Module-level defaults below

    export TSHEPO_MODELS_DIR=/opt/tshepo/models
    export TSHEPO_LOG_LEVEL=DEBUG
    export TSHEPO_RATE_MAX=60
"""
from __future__ import annotations

import json
import os
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR: str = os.path.join(ROOT, "data")
MODELS_DIR: str = os.path.join(ROOT, "models")
DOCS_DIR: str = os.path.join(ROOT, "docs")

LABELS: list[str] = ["genuine_transaction", "suspicious", "noise"]
L2I: dict[str, int] = {l: i for i, l in enumerate(LABELS)}

# --- runtime knobs (overridable via TSHEPO_* env vars) ---
LOG_LEVEL: str = os.environ.get("TSHEPO_LOG_LEVEL", "INFO")
JSON_LOGS: bool = os.environ.get("TSHEPO_JSON_LOGS", "0") == "1"
MAX_BODY_LEN: int = int(os.environ.get("TSHEPO_MAX_BODY_LEN", "4096"))
MAX_SENDER_LEN: int = int(os.environ.get("TSHEPO_MAX_SENDER_LEN", "64"))
MAX_RECORDS_PER_USER: int = int(os.environ.get("TSHEPO_MAX_RECORDS_PER_USER", "5000"))
MAX_RECORDS_TOTAL: int = int(os.environ.get("TSHEPO_MAX_RECORDS_TOTAL", "200000"))
RATE_WINDOW: float = float(os.environ.get("TSHEPO_RATE_WINDOW", "60"))
RATE_MAX: int = int(os.environ.get("TSHEPO_RATE_MAX", "120"))
API_KEY: str | None = os.environ.get("TSHEPO_API_KEY")
ENABLE_DOCS: bool = os.environ.get("TSHEPO_ENABLE_DOCS", "0") == "1"
CORS_ORIGINS: list[str] = [o.strip() for o in os.environ.get("TSHEPO_CORS_ORIGINS", "").split(",") if o.strip()]
MAX_REQUEST_SIZE_MB: float = float(os.environ.get("TSHEPO_MAX_REQUEST_SIZE_MB", "10"))

# --- consent, retention and data-quality policy -------------------------------
# Where the consent ledger and the collected messages live. Mount this volume
# read-write; everything else in the container can stay read-only.
DB_PATH: str = os.environ.get("TSHEPO_DB_PATH", os.path.join(DATA_DIR, "tshepo.db"))
# How long a consent stays valid before the data behind it must be purged.
CONSENT_TTL_DAYS: int = int(os.environ.get("TSHEPO_CONSENT_TTL_DAYS", "90"))
# Hard retention limit for collected messages, independent of the consent.
RETENTION_DAYS: int = int(os.environ.get("TSHEPO_RETENTION_DAYS", "180"))
# Share of balance-chain links that must hold before a score is presented as a
# decision. Below this the passport is referred for manual review.
MIN_INTEGRITY: float = float(os.environ.get("TSHEPO_MIN_INTEGRITY", "0.9"))
# Share of the 13-week window the history must cover.
MIN_WINDOW_COVERAGE: float = float(os.environ.get("TSHEPO_MIN_WINDOW_COVERAGE", "0.25"))
# Withhold the numeric score whenever the passport is not "scored".
WITHHOLD_SCORE: bool = os.environ.get("TSHEPO_WITHHOLD_SCORE", "1") == "1"
# Serve a score model that was fitted on synthetic outcomes. Off by default:
# the shipped model is fitted on synthetic labels and proves nothing about
# real repayment behaviour.
ALLOW_UNCALIBRATED_MODEL: bool = os.environ.get("TSHEPO_ALLOW_UNCALIBRATED_MODEL", "0") == "1"
# Serve the web capture portal.
ENABLE_PORTAL: bool = os.environ.get("TSHEPO_ENABLE_PORTAL", "1") == "1"

_CONFIG_CACHE: dict[str, Any] | None = None


def _load_file_config() -> dict[str, Any]:
    """Load tshepo.yaml or tshepo.json from cwd, then repo root."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    candidates = [
        os.path.join(os.getcwd(), "tshepo.yaml"),
        os.path.join(os.getcwd(), "tshepo.json"),
        os.path.join(ROOT, "tshepo.yaml"),
        os.path.join(ROOT, "tshepo.json"),
    ]
    cfg: dict[str, Any] = {}
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            if path.endswith(".json"):
                with open(path, encoding="utf-8") as fh:
                    cfg = json.load(fh)
            elif path.endswith(".yaml"):
                try:
                    import yaml
                except ImportError:
                    continue
                with open(path, encoding="utf-8") as fh:
                    cfg = yaml.safe_load(fh) or {}
        except Exception:  # noqa: BLE001 - config file issues are non-fatal
            cfg = {}
        break
    _CONFIG_CACHE = cfg if isinstance(cfg, dict) else {}
    return _CONFIG_CACHE


def get(key: str, default: Any = None) -> Any:
    """Resolve a setting: env var > config file > module default > default arg."""
    env_val = os.environ.get(f"TSHEPO_{key.upper()}")
    if env_val is not None:
        return env_val
    file_cfg = _load_file_config()
    if key in file_cfg:
        return file_cfg[key]
    return globals().get(key, default)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def model_path(name: str) -> str:
    """Resolve a model path: absolute paths are used as-is, "models/x" is relative to the
    repo root, anything else is taken as a file name inside models/."""
    if os.path.isabs(name):
        return name
    if name.replace("\\", "/").startswith("models/"):
        return os.path.join(ROOT, name)
    return os.path.join(MODELS_DIR, name)
