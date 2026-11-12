import json
from . import config

_CATALOG = None


def load():
    global _CATALOG
    if _CATALOG is None:
        items = json.load(open(config.DATA_DIR / "catalog.json", encoding="utf8"))
        ov = config.DATA_DIR / "image_overrides.json"  # written by scripts/refresh_images.py (Pexels)
        if ov.exists():
            o = json.load(open(ov, encoding="utf8"))
            for p in items:
                if p["id"] in o:
                    p["image"] = o[p["id"]]
        _CATALOG = items
    return _CATALOG


def get(pid: str):
    return next((p for p in load() if p["id"] == pid), None)
