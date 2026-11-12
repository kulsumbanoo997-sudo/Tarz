"""Image storage: local /uploads by default, Cloudinary mirror when CLOUDINARY_URL is set."""
import hashlib
import re
import time
import uuid

import httpx

from . import config


def save_local(data: bytes, prefix: str, ext: str = "jpg") -> tuple[str, str]:
    name = f"{prefix}_{uuid.uuid4().hex[:12]}.{ext}"
    path = config.UPLOAD_DIR / name
    path.write_bytes(data)
    return str(path), f"/uploads/{name}"


def cloudinary_upload(data: bytes, public_id: str) -> str | None:
    """Signed upload via REST. Returns secure_url or None (not configured / failed)."""
    m = re.match(r"cloudinary://([^:]+):([^@]+)@(.+)", config.CLOUDINARY_URL or "")
    if not m:
        return None
    key, secret, cloud = m.groups()
    ts = str(int(time.time()))
    to_sign = f"public_id={public_id}&timestamp={ts}{secret}"
    sig = hashlib.sha1(to_sign.encode()).hexdigest()
    try:
        r = httpx.post(f"https://api.cloudinary.com/v1_1/{cloud}/image/upload", timeout=30,
                       data={"api_key": key, "timestamp": ts, "public_id": public_id, "signature": sig},
                       files={"file": ("img.jpg", data)})
        return r.json().get("secure_url") if r.status_code == 200 else None
    except Exception:
        return None
