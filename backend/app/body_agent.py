"""Body Analysis Agent: photo or manual input -> interpretable body profile.

Pipeline (photo): MediaPipe pose landmarks -> widths (cm, scaled by height)
-> body shape (shoulder/hip ratio), skin tone (colour patches), size + fit confidence.
All thresholds are hand-set heuristics. They are NOT calibrated on labelled data.
"""
import io
import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

SIZES = ["S", "M", "L", "XL", "XXL"]

# Generic placeholder size charts in cm (bust/chest, waist, hip). Replace with the shop's real charts.
SIZE_CHARTS = {
    "Women": {
        "S": {"bust": (82, 88), "waist": (64, 70), "hip": (88, 94)},
        "M": {"bust": (88, 94), "waist": (70, 76), "hip": (94, 100)},
        "L": {"bust": (94, 100), "waist": (76, 82), "hip": (100, 106)},
        "XL": {"bust": (100, 106), "waist": (82, 88), "hip": (106, 112)},
        "XXL": {"bust": (106, 112), "waist": (88, 94), "hip": (112, 118)},
    },
    "Men": {
        "S": {"bust": (86, 92), "waist": (72, 78), "hip": (90, 96)},
        "M": {"bust": (92, 98), "waist": (78, 84), "hip": (96, 102)},
        "L": {"bust": (98, 104), "waist": (84, 90), "hip": (102, 108)},
        "XL": {"bust": (104, 110), "waist": (90, 96), "hip": (108, 114)},
        "XXL": {"bust": (110, 116), "waist": (96, 102), "hip": (114, 120)},
    },
}

# Uncalibrated geometry constants used to turn 2-D landmark widths into circumferences.
CAL = {
    "nose_to_ankle_frac": 0.885,  # (nose -> ankle) / stature
    "chest_width_frac": 0.90,     # chest breadth / shoulder joint distance
    "chest_depth_ratio": 0.66,    # chest depth / chest breadth
    "hip_body_factor": 1.20,      # body hip breadth / hip-joint distance
    "hip_depth_ratio": 0.70,      # hip depth / hip breadth
}


class AnalysisError(Exception):
    pass


# ---------------------------------------------------------------- rules
def classify_body_shape(rho: float) -> str:
    if rho > 1.20:
        return "Inverted Triangle"
    if rho > 1.06:
        return "Apple"
    if rho < 0.82:
        return "Pear"
    if rho < 0.94:
        return "Rectangle"
    return "Hourglass"


def body_shape_from_measurements(bust: float, waist: Optional[float], hip: float) -> str:
    """Manual-mode rule using circumferences (differs from the photo rule, which only sees widths)."""
    diff = (bust - hip) / hip
    if diff > 0.05:
        return "Inverted Triangle"
    if diff < -0.05:
        return "Pear"
    if waist is None:
        return "Rectangle"
    w = waist / ((bust + hip) / 2)
    if w <= 0.75:
        return "Hourglass"
    if w >= 0.90:
        return "Apple"
    return "Rectangle"


def classify_skin(rgb) -> dict:
    r, g, b = [float(x) for x in rgb]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    depth = "Fair" if y > 170 else "Medium" if y > 110 else "Deep"
    gb = g - b
    undertone = "warm" if gb > 5 else "cool" if gb < -1 else "neutral"
    return {"depth": depth, "undertone": undertone, "luminance": round(y, 1), "g_minus_b": round(gb, 1)}


def _ellipse_perimeter(width: float, depth: float) -> float:
    a, b = width / 2, depth / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def size_distances(measurements: dict, gender: str, size: str) -> dict:
    chart = SIZE_CHARTS[gender][size]
    out = {}
    for m, val in measurements.items():
        if m in chart and val:
            lo, hi = chart[m]
            out[m] = 0.0 if lo <= val <= hi else (lo - val if val < lo else val - hi) / (hi - lo)
    return out


def fit_confidence(distances: dict) -> float:
    """c = 1 - mean_m min(1, d_m)   (Eq. c in the paper)."""
    if not distances:
        return 0.0
    return round(1 - sum(min(1.0, d) for d in distances.values()) / len(distances), 3)


def recommend_size(measurements: dict, gender: str) -> dict:
    per = {}
    for s in SIZES:
        d = size_distances(measurements, gender, s)
        per[s] = {"confidence": fit_confidence(d), "mean_distance": round(sum(d.values()) / len(d), 3) if d else None}
    usable = {s: v for s, v in per.items() if v["mean_distance"] is not None}
    if not usable:
        return {"recommended": None, "fit_confidence": 0.0, "per_size": per}
    best = min(usable, key=lambda s: (usable[s]["mean_distance"], SIZES.index(s)))
    return {"recommended": best, "fit_confidence": usable[best]["confidence"], "per_size": per}


# ---------------------------------------------------------------- photo
def load_image(data: bytes, max_side: int = 1024) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise AnalysisError("Could not read the image file.") from e
    img.thumbnail((max_side, max_side))
    return np.array(img)


def _skin_from_pose(img: np.ndarray, lm) -> Optional[dict]:
    h, w = img.shape[:2]
    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])
    le, re_ = P(2), P(5)
    e = float(np.linalg.norm(le - re_))
    if e < 8 or min(lm[2].visibility, lm[5].visibility) < 0.5:
        return None
    theta = math.atan2(re_[1] - le[1], re_[0] - le[0])
    down = np.array([-math.sin(theta), math.cos(theta)])
    mid = (le + re_) / 2
    centers = [mid - down * 0.9 * e, le + down * 0.85 * e, re_ + down * 0.85 * e]
    half = max(2, int(0.18 * e))
    px = []
    for c in centers:
        x, y = int(c[0]), int(c[1])
        x0, x1, y0, y1 = max(0, x - half), min(w, x + half), max(0, y - half), min(h, y + half)
        if x1 > x0 and y1 > y0:
            px.append(img[y0:y1, x0:x1].reshape(-1, 3))
    if not px:
        return None
    rgb = np.median(np.concatenate(px, axis=0), axis=0)
    out = classify_skin(rgb)
    out["rgb"] = [int(v) for v in rgb]
    return out


def analyze_photo(data: bytes, gender: str, height_cm: Optional[float]) -> dict:
    import mediapipe as mp  # imported lazily: heavy

    img = load_image(data)
    h, w = img.shape[:2]
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        res = pose.process(img)
    if not res.pose_landmarks:
        raise AnalysisError("No person detected. Use a clear, full-body, front-facing photo.")
    lm = res.pose_landmarks.landmark
    L = mp.solutions.pose.PoseLandmark
    need = [L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP]
    if any(lm[i].visibility < 0.3 for i in need):
        raise AnalysisError("Shoulders and hips are not clearly visible. Stand straight, facing the camera.")

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    sh_px = float(np.linalg.norm(P(L.LEFT_SHOULDER) - P(L.RIGHT_SHOULDER)))
    hip_px = float(np.linalg.norm(P(L.LEFT_HIP) - P(L.RIGHT_HIP)))
    if hip_px < 1:
        raise AnalysisError("Could not measure the hips. Try another photo.")
    rho = sh_px / hip_px
    warnings = []
    prof = {
        "gender": gender, "source": "photo", "height_cm": height_cm,
        "rho": round(rho, 3), "body_shape": classify_body_shape(rho),
        "measurements": {}, "approximate": True, "warnings": warnings,
        "skin": _skin_from_pose(img, lm),
    }
    if prof["skin"] is None:
        warnings.append("Face not clear enough to estimate skin tone; please choose it manually.")

    ankles_ok = min(lm[L.LEFT_ANKLE].visibility, lm[L.RIGHT_ANKLE].visibility) > 0.4 and lm[L.NOSE].visibility > 0.4
    if height_cm and ankles_ok:
        ank_y = (P(L.LEFT_ANKLE)[1] + P(L.RIGHT_ANKLE)[1]) / 2
        span = ank_y - P(L.NOSE)[1]
        if span > 50:
            cm_per_px = height_cm / (span / CAL["nose_to_ankle_frac"])
            sh_cm, hip_cm = sh_px * cm_per_px, hip_px * cm_per_px
            chest_w = sh_cm * CAL["chest_width_frac"]
            hip_w = hip_cm * CAL["hip_body_factor"]
            prof["measurements"] = {
                "shoulder_width_cm": round(sh_cm, 1), "hip_width_cm": round(hip_w, 1),
                "bust": round(_ellipse_perimeter(chest_w, chest_w * CAL["chest_depth_ratio"]), 1),
                "hip": round(_ellipse_perimeter(hip_w, hip_w * CAL["hip_depth_ratio"]), 1),
            }
            warnings.append("Bust and hip are estimates from one 2-D photo; enter tape measurements for accuracy. Waist is not estimated.")
    else:
        warnings.append("Full body (head to ankles) and height are needed to estimate measurements.")
    return prof


# ---------------------------------------------------------------- manual
def analyze_manual(gender, bust, waist, hip, height_cm=None, body_shape=None, undertone="neutral", depth="Medium") -> dict:
    if not (bust and hip):
        raise AnalysisError("Please enter at least bust/chest and hip measurements (cm).")
    for name, v in (("bust", bust), ("hip", hip), ("waist", waist)):
        if v is not None and not (40 <= v <= 200):
            raise AnalysisError(f"{name} must be between 40 and 200 cm.")
    meas = {"bust": bust, "hip": hip}
    if waist:
        meas["waist"] = waist
    return {
        "gender": gender, "source": "manual", "height_cm": height_cm, "rho": None,
        "body_shape": body_shape or body_shape_from_measurements(bust, waist, hip),
        "measurements": meas, "approximate": False, "warnings": [],
        "skin": {"depth": depth, "undertone": undertone, "luminance": None, "g_minus_b": None},
    }


def finish_profile(prof: dict) -> dict:
    meas = {k: v for k, v in prof["measurements"].items() if k in ("bust", "waist", "hip")}
    prof["size"] = recommend_size(meas, prof["gender"]) if meas else {"recommended": None, "fit_confidence": 0.0, "per_size": {}}
    return prof
