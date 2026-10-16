"""SSPO and SSPO-Bench package."""
from __future__ import annotations

import os

if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

__version__ = "1.0.0"
