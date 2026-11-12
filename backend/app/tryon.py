"""Virtual try-on orchestration: smart garment extraction, ordered backends, per-stage time-outs."""
import asyncio
import io
import shutil
import uuid

import httpx
import numpy as np
from PIL import Image

from . import config

STAGE_TIMEOUT = config.T_CONNECT + config.T_PREDICT


def extract_garment_region(img: Image.Image, category: str = "") -> Image.Image:
    """If the garment photo contains a full-body or half-body model, extract the garment area.
    This eliminates model faces, background clutter (plants, bags, walls), shoes, and floors
    that ruin virtual try-on texture transfer.
    """
    try:
        import mediapipe as mp
        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            res = pose.process(arr)
            if not res or not res.pose_landmarks:
                return img
            lm = res.pose_landmarks.landmark
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            if l_sh.visibility < 0.3 or r_sh.visibility < 0.3:
                return img

            sh_y = min(l_sh.y, r_sh.y) * h
            hip_y = max(l_hip.y, r_hip.y) * h
            torso_h = max(20.0, hip_y - sh_y)

            # Neck / collar headroom
            y_top = max(0, int(sh_y - 0.22 * torso_h))

            # Bottom cutoff based on garment type
            cat = (category or "").lower()
            if any(k in cat for k in ("dress", "kurti", "saree", "anarkali", "gown")):
                y_bottom = min(h, int(hip_y + 0.65 * torso_h))
            else:  # Tops, shirts, jackets
                y_bottom = min(h, int(hip_y + 0.25 * torso_h))

            # Lateral bounds: shoulders + elbows ONLY (exclude wrists and hands to prevent transferring model's arms)
            xs = [l_sh.x, r_sh.x]
            for pt in [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]:
                if lm[pt].visibility > 0.4:
                    xs.append(lm[pt].x)
            x_min = max(0, int((min(xs) - 0.05) * w))
            x_max = min(w, int((max(xs) + 0.05) * w))

            if (x_max - x_min) > 80 and (y_bottom - y_top) > 80:
                return img.crop((x_min, y_top, x_max, y_bottom))
    except Exception:
        pass
    return img


def letterbox(data: bytes, category: str = "", w: int = 768, h: int = 1024) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # Auto-extract garment from model photo to avoid background/model contamination
    img = extract_garment_region(img, category)
    s = min(w / img.width, h / img.height)
    new_w, new_h = max(1, int(img.width * s)), max(1, int(img.height * s))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(img, ((w - new_w) // 2, (h - new_h) // 2))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=95)
    return buf.getvalue()


async def prepare_garment(url: str, category: str = "") -> str:
    """Download + smartly crop + letterbox the garment image."""
    try:
        async with httpx.AsyncClient(timeout=config.T_GARMENT, follow_redirects=True) as cx:
            r = await cx.get(url)
            r.raise_for_status()
        path = config.UPLOAD_DIR / f"garment_{uuid.uuid4().hex[:10]}.jpg"
        path.write_bytes(letterbox(r.content, category=category))
        return str(path)
    except Exception:
        return url


def _hf_space(space: str, person: str, garment: str, description: str, use_token: bool = True) -> str:
    import inspect
    from gradio_client import Client, handle_file
    kw = {"httpx_kwargs": {"timeout": 120.0}}
    if use_token and config.HF_TOKEN:
        param = "token" if "token" in inspect.signature(Client.__init__).parameters else "hf_token"
        kw[param] = config.HF_TOKEN
    client = Client(space, verbose=False, **kw)
    try:
        res = client.predict(
            {"background": handle_file(person), "layers": [], "composite": None},
            handle_file(garment),
            description,
            True,   # is_checked (use auto-masking on person)
            True,   # is_checked_crop (maintains 3:4 aspect ratio, prevents body squeezing & extra limbs)
            30,     # denoise_steps
            42,     # seed
            api_name="/tryon"
        )
    except Exception as e:
        # If the user's HF_TOKEN exceeded the daily ZeroGPU quota, seamlessly fallback to anonymous
        if use_token and config.HF_TOKEN and ("quota" in str(e).lower() or "exceeded" in str(e).lower()):
            return _hf_space(space, person, garment, description, use_token=False)
        raise
    out = res[0] if isinstance(res, (list, tuple)) else res
    if isinstance(out, dict):
        out = out.get("path") or out.get("url")
    if not out:
        raise RuntimeError("model returned no image")
    return str(out)


def _replicate(person: str, garment: str, name: str, category: str) -> bytes:
    import replicate
    cl = replicate.Client(api_token=config.REPLICATE_API_TOKEN)
    cat = "dresses" if category in ("Dresses", "Sarees") else "upper_body"
    with open(person, "rb") as hp, open(garment, "rb") as gp:
        out = cl.run("cuuupid/idm-vton", input={"human_img": hp, "garm_img": gp, "garment_des": name, "category": cat})
    if isinstance(out, (list, tuple)):
        out = out[0]
    if hasattr(out, "read"):
        return out.read()
    r = httpx.get(str(out), timeout=60)
    r.raise_for_status()
    return r.content


async def run_tryon(person_path: str, product: dict) -> dict:
    garment = await prepare_garment(product["image"], product.get("category", ""))
    
    # Build rich prompt for high-fidelity diffusion conditioning
    styles = product.get("styles", [])
    styles_str = ", ".join(styles) if isinstance(styles, list) else str(styles)
    prompt_parts = [
        product.get("name", "clothing"),
        product.get("colorFamily", ""),
        styles_str,
        product.get("category", ""),
        "detailed authentic fabric pattern, sharp print texture, realistic clothing fit"
    ]
    description = " ".join([p for p in prompt_parts if p]).strip()

    backends = [
        ("IDM-VTON (HF Space)", lambda: _hf_space("yisol/IDM-VTON", person_path, garment, description)),
        ("Virtual-Try-On (HF Space)", lambda: _hf_space("Nymbo/Virtual-Try-On", person_path, garment, description)),
    ]
    errors = []
    for name, fn in backends:
        try:
            src = await asyncio.wait_for(asyncio.to_thread(fn), STAGE_TIMEOUT)
            dest = config.UPLOAD_DIR / f"tryon_{uuid.uuid4().hex[:12]}.png"
            shutil.copy(src, dest)
            return {"backend": name, "path": str(dest), "url": f"/uploads/{dest.name}", "errors": errors}
        except Exception as e:
            errors.append(f"{name}: {str(e)[:160] or type(e).__name__}")
    if config.REPLICATE_API_TOKEN:
        try:
            data = await asyncio.wait_for(asyncio.to_thread(_replicate, person_path, garment, description, product.get("category", "")), STAGE_TIMEOUT)
            dest = config.UPLOAD_DIR / f"tryon_{uuid.uuid4().hex[:12]}.png"
            dest.write_bytes(data)
            return {"backend": "IDM-VTON (Replicate)", "path": str(dest), "url": f"/uploads/{dest.name}", "errors": errors}
        except Exception as e:
            errors.append(f"Replicate: {str(e)[:160] or type(e).__name__}")
    raise RuntimeError("All try-on backends failed. " + " | ".join(errors))
