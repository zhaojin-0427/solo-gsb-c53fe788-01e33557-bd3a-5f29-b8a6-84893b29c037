"""Application configuration and shared singletons."""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageCms

from . import lcms_native
from .profiles import ProfileManager

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SPD_DATA_DIR", BASE_DIR.parent / "data"))
PROFILE_DIR = DATA_DIR / "profiles"
JOB_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "softproof.db"

MAX_DIMENSION = int(os.environ.get("SPD_MAX_DIMENSION", "3000"))
MAX_UPLOAD_MB = int(os.environ.get("SPD_MAX_UPLOAD_MB", "50"))


@dataclass
class AppState:
    pm: ProfileManager
    lab_profile: ImageCms.ImageCmsProfile
    srgb_profile: ImageCms.ImageCmsProfile
    lab_bytes: bytes
    srgb_bytes: bytes


_state: AppState | None = None


def init_state() -> AppState:
    global _state
    if _state is None:
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        pm = ProfileManager(PROFILE_DIR)
        lab_bytes = lcms_native.build_lab_profile_bytes()
        srgb_bytes = lcms_native.build_srgb_profile_bytes()
        lab_profile = ImageCms.ImageCmsProfile(io.BytesIO(lab_bytes))
        srgb_profile = ImageCms.ImageCmsProfile(io.BytesIO(srgb_bytes))
        _state = AppState(
            pm=pm,
            lab_profile=lab_profile,
            srgb_profile=srgb_profile,
            lab_bytes=lab_bytes,
            srgb_bytes=srgb_bytes,
        )
    return _state
