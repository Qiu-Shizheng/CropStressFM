from __future__ import annotations

import json
from pathlib import Path


def weights_directory() -> Path:
    return Path(__file__).resolve().parent / "weights"


def load_config() -> dict[str, object]:
    return json.loads((weights_directory() / "config.json").read_text(encoding="utf-8"))
