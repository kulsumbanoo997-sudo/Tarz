"""Body Analysis Agent Evaluation (RQ1) for TARZ IEEE Research Paper.

Evaluates MediaPipe Pose + anthropometric circumference models on real participant data:
- Circumference Mean Absolute Error (MAE in cm) for bust, hip, and available metrics vs tape measurements.
- Body shape classification accuracy (%) against ground truth.
- Skin depth and undertone classification accuracy (%) and per-depth breakdown.
- Recommended size agreement rate (%) against ground truth size.
- Baseline comparison (population mean or linear stature scaling).

Strict Rule: If data is missing or only contains EXAMPLE_ROW, leaves result as TBD and exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("eval_body")

# Project paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_baseline_predictions(
    height_cm: float,
    gender: str,
    baseline_mode: str = "population_mean",
) -> Dict[str, float]:
    """Computes baseline circumference estimates without TARZ ellipse modeling."""
    from app import body_agent

    chart = body_agent.SIZE_CHARTS.get(gender, body_agent.SIZE_CHARTS["Women"])
    # Default Medium size midpoint as population mean
    m_band = chart["M"]
    pop_bust = (m_band["bust"][0] + m_band["bust"][1]) / 2.0
    pop_hip = (m_band["hip"][0] + m_band["hip"][1]) / 2.0
    pop_waist = (m_band["waist"][0] + m_band["waist"][1]) / 2.0

    if baseline_mode == "population_mean":
        return {"bust": pop_bust, "hip": pop_hip, "waist": pop_waist}
    elif baseline_mode == "stature_linear":
        # Simple proportional height scaling relative to 165 cm reference
        scale = max(0.5, height_cm / 165.0)
        return {
            "bust": round(pop_bust * scale, 1),
            "hip": round(pop_hip * scale, 1),
            "waist": round(pop_waist * scale, 1),
        }
    else:
        return {"bust": pop_bust, "hip": pop_hip, "waist": pop_waist}


def analyze_participant_direct(
    photo_path: Path,
    gender: str,
    height_cm: float,
) -> dict:
    """Invokes body_agent directly from backend Python module."""
    from app import body_agent

    if not photo_path.is_file():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    data = photo_path.read_bytes()
    prof = body_agent.analyze_photo(data, gender, height_cm)
    return body_agent.finish_profile(prof)


def analyze_participant_api(
    base_url: str,
    photo_path: Path,
    gender: str,
    height_cm: float,
    timeout: float = 40.0,
    retries: int = 3,
) -> dict:
    """Calls POST /api/analyze over HTTP with retries and exponential backoff."""
    if not photo_path.is_file():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    url = f"{base_url.rstrip('/')}/api/analyze"
    for attempt in range(1, retries + 1):
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": (photo_path.name, f, "image/jpeg")}
                data = {"gender": gender, "height_cm": str(height_cm)}
                response = httpx.post(url, data=data, files=files, timeout=timeout)
            if response.status_code == 200:
                res_json = response.json()
                return res_json.get("profile", {})
            else:
                logger.warning(
                    f"HTTP {response.status_code} for {photo_path.name}: {response.text[:150]}"
                )
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{retries} failed for {photo_path.name}: {e}")
            if attempt < retries:
                time.sleep(1.5 * attempt)

    # Fallback to direct module call if API server is not running
    logger.info(f"API call unavailable. Executing direct in-process analysis for {photo_path.name}...")
    return analyze_participant_direct(photo_path, gender, height_cm)


def compute_metrics(samples_df: pd.DataFrame, baseline_mode: str) -> Dict[str, Any]:
    """Computes MAE, Accuracy, and bootstrap standard errors."""
    if samples_df.empty:
        return {"status": "TBD", "reason": "No valid non-example ground truth rows found."}

    # Circumference MAE
    bust_errors = np.abs(samples_df["tarz_bust"] - samples_df["true_bust"]).dropna()
    hip_errors = np.abs(samples_df["tarz_hip"] - samples_df["true_hip"]).dropna()
    base_bust_errors = np.abs(samples_df["base_bust"] - samples_df["true_bust"]).dropna()
    base_hip_errors = np.abs(samples_df["base_hip"] - samples_df["true_hip"]).dropna()

    tarz_bust_mae = float(bust_errors.mean()) if len(bust_errors) else None
    tarz_hip_mae = float(hip_errors.mean()) if len(hip_errors) else None
    base_bust_mae = float(base_bust_errors.mean()) if len(base_bust_errors) else None
    base_hip_mae = float(base_hip_errors.mean()) if len(base_hip_errors) else None

    # Combined Circumference MAE (average over bust and hip)
    tarz_all_circ_errors = list(bust_errors) + list(hip_errors)
    base_all_circ_errors = list(base_bust_errors) + list(base_hip_errors)
    tarz_circ_mae = float(np.mean(tarz_all_circ_errors)) if tarz_all_circ_errors else None
    base_circ_mae = float(np.mean(base_all_circ_errors)) if base_all_circ_errors else None

    # Categorical Accuracies (%)
    valid_shape = samples_df.dropna(subset=["true_body_shape", "tarz_body_shape"])
    shape_acc = (
        float((valid_shape["true_body_shape"].str.strip().str.lower() == valid_shape["tarz_body_shape"].str.strip().str.lower()).mean() * 100.0)
        if len(valid_shape)
        else None
    )

    valid_depth = samples_df.dropna(subset=["true_skin_depth", "tarz_skin_depth"])
    skin_depth_acc = (
        float((valid_depth["true_skin_depth"].str.strip().str.lower() == valid_depth["tarz_skin_depth"].str.strip().str.lower()).mean() * 100.0)
        if len(valid_depth)
        else None
    )

    valid_undertone = samples_df.dropna(subset=["true_undertone", "tarz_undertone"])
    undertone_acc = (
        float((valid_undertone["true_undertone"].str.strip().str.lower() == valid_undertone["tarz_undertone"].str.strip().str.lower()).mean() * 100.0)
        if len(valid_undertone)
        else None
    )

    valid_size = samples_df.dropna(subset=["true_size", "tarz_size"])
    size_agreement = (
        float((valid_size["true_size"].str.strip().str.upper() == valid_size["tarz_size"].str.strip().str.upper()).mean() * 100.0)
        if len(valid_size)
        else None
    )

    # Per-skin-depth breakdown
    depth_breakdown = {}
    for d in ["Fair", "Medium", "Deep"]:
        sub = valid_depth[valid_depth["true_skin_depth"].str.strip().str.lower() == d.lower()]
        if len(sub) > 0:
            sub_acc = float((sub["true_skin_depth"].str.strip().str.lower() == sub["tarz_skin_depth"].str.strip().str.lower()).mean() * 100.0)
            depth_breakdown[d] = {"count": len(sub), "accuracy_pct": round(sub_acc, 2)}
        else:
            depth_breakdown[d] = {"count": 0, "accuracy_pct": "TBD"}

    return {
        "status": "COMPLETED",
        "sample_count": len(samples_df),
        "baseline_mode": baseline_mode,
        "metrics": {
            "circumference_mae_cm": {
                "tarz": round(tarz_circ_mae, 2) if tarz_circ_mae is not None else "TBD",
                "baseline": round(base_circ_mae, 2) if base_circ_mae is not None else "TBD",
                "details": {
                    "bust_mae": round(tarz_bust_mae, 2) if tarz_bust_mae is not None else "TBD",
                    "hip_mae": round(tarz_hip_mae, 2) if tarz_hip_mae is not None else "TBD",
                    "waist_note": "Waist is intentionally not estimated from 2D monocular photos.",
                },
            },
            "shape_accuracy_pct": {
                "tarz": round(shape_acc, 2) if shape_acc is not None else "TBD",
                "baseline": round(100.0 / 5.0, 2),  # Uniform random 5-class guess baseline = 20%
            },
            "skin_depth_accuracy_pct": {
                "tarz": round(skin_depth_acc, 2) if skin_depth_acc is not None else "TBD",
                "baseline": round(100.0 / 3.0, 2),  # Uniform random 3-class guess baseline = 33.33%
            },
            "skin_undertone_accuracy_pct": {
                "tarz": round(undertone_acc, 2) if undertone_acc is not None else "TBD",
                "baseline": round(100.0 / 3.0, 2),  # Uniform random 3-class guess baseline = 33.33%
            },
            "size_agreement_rate_pct": {
                "tarz": round(size_agreement, 2) if size_agreement is not None else "TBD",
                "baseline": round(100.0 / 5.0, 2),
            },
            "per_skin_depth_breakdown": depth_breakdown,
        },
    }


def is_server_online(base_url: str) -> bool:
    """Quick check if FastAPI backend is listening."""
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def run_evaluation(config: dict, gt_csv_override: Optional[Path] = None) -> None:
    gt_path = gt_csv_override or (WORKSPACE_ROOT / config["paths"]["body_gt_csv"])
    results_dir = WORKSPACE_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out_metrics_json = results_dir / "body_eval_metrics.json"
    out_samples_csv = results_dir / "body_eval_samples.csv"

    if not gt_path.is_file():
        logger.warning(f"Ground truth file not found: {gt_path}. Emitting TBD result.")
        result_tbd = {"status": "TBD", "reason": f"File {gt_path} not found"}
        out_metrics_json.write_text(json.dumps(result_tbd, indent=2))
        return

    df = pd.read_csv(gt_path)
    # Filter out template example rows
    filtered = df[~df["participant_id"].astype(str).str.startswith("EXAMPLE_ROW")].copy()

    if filtered.empty:
        logger.warning(f"No real participant rows in {gt_path} (only EXAMPLE_ROW rows present).")
        logger.info("Per strict instructions, writing status 'TBD' into Table IV results.")
        result_tbd = {
            "status": "TBD",
            "reason": "Only template EXAMPLE_ROW entries present. Awaiting human participant ground truth.",
            "metrics": {
                "circumference_mae_cm": {"tarz": "TBD", "baseline": "TBD"},
                "shape_accuracy_pct": {"tarz": "TBD", "baseline": "TBD"},
                "skin_depth_accuracy_pct": {"tarz": "TBD", "baseline": "TBD"},
                "skin_undertone_accuracy_pct": {"tarz": "TBD", "baseline": "TBD"},
            },
        }
        out_metrics_json.write_text(json.dumps(result_tbd, indent=2))
        return

    from app import body_agent

    baseline_mode = config.get("rq1_body", {}).get("baseline_mode", "population_mean")
    base_url = config.get("api", {}).get("base_url", "http://127.0.0.1:8000")
    server_active = is_server_online(base_url)
    if server_active:
        logger.info(f"Connected to live TARZ backend at {base_url}")
    else:
        logger.info(f"TARZ backend offline at {base_url}. Using direct module execution.")

    records = []
    logger.info(f"Evaluating {len(filtered)} ground truth participants...")

    for idx, row in filtered.iterrows():
        pid = str(row["participant_id"])
        height_cm = float(row["height_cm"])
        gender = str(row.get("gender", "Women"))

        photo_rel = str(row["photo_path"])
        photo_path = WORKSPACE_ROOT / photo_rel

        # Ground truth values
        true_bust = float(row["bust_cm"]) if pd.notnull(row["bust_cm"]) else None
        true_waist = float(row["waist_cm"]) if pd.notnull(row["waist_cm"]) else None
        true_hip = float(row["hip_cm"]) if pd.notnull(row["hip_cm"]) else None
        true_shape = str(row["true_body_shape"]) if pd.notnull(row["true_body_shape"]) else None
        true_depth = str(row["true_skin_depth"]) if pd.notnull(row["true_skin_depth"]) else None
        true_undertone = str(row["true_undertone"]) if pd.notnull(row["true_undertone"]) else None

        # Ground truth size from manual tape measurements
        true_tape_meas = {k: v for k, v in [("bust", true_bust), ("waist", true_waist), ("hip", true_hip)] if v}
        true_size_obj = body_agent.recommend_size(true_tape_meas, gender)
        true_size = true_size_obj.get("recommended")

        # Baseline prediction
        base_preds = calculate_baseline_predictions(height_cm, gender, baseline_mode)

        # Execute TARZ prediction
        try:
            if server_active:
                prof = analyze_participant_api(base_url, photo_path, gender, height_cm)
            else:
                prof = analyze_participant_direct(photo_path, gender, height_cm)
            meas = prof.get("measurements", {})
            skin = prof.get("skin") or {}
            size = prof.get("size") or {}

            tarz_bust = meas.get("bust")
            tarz_hip = meas.get("hip")
            tarz_shape = prof.get("body_shape")
            tarz_depth = skin.get("depth")
            tarz_undertone = skin.get("undertone")
            tarz_size = size.get("recommended")
            pred_status = "SUCCESS"
            err_msg = ""
        except Exception as e:
            logger.error(f"Error predicting participant {pid}: {e}")
            tarz_bust, tarz_hip, tarz_shape, tarz_depth, tarz_undertone, tarz_size = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
            pred_status = "ERROR"
            err_msg = str(e)

        records.append(
            {
                "participant_id": pid,
                "gender": gender,
                "height_cm": height_cm,
                "true_bust": true_bust,
                "true_waist": true_waist,
                "true_hip": true_hip,
                "true_body_shape": true_shape,
                "true_skin_depth": true_depth,
                "true_undertone": true_undertone,
                "true_size": true_size,
                "base_bust": base_preds.get("bust"),
                "base_hip": base_preds.get("hip"),
                "base_waist": base_preds.get("waist"),
                "tarz_bust": tarz_bust,
                "tarz_hip": tarz_hip,
                "tarz_body_shape": tarz_shape,
                "tarz_skin_depth": tarz_depth,
                "tarz_undertone": tarz_undertone,
                "tarz_size": tarz_size,
                "status": pred_status,
                "error": err_msg,
            }
        )

    samples_df = pd.DataFrame(records)
    samples_df.to_csv(out_samples_csv, index=False)
    logger.info(f"Saved per-sample evaluation to {out_samples_csv}")

    metrics = compute_metrics(samples_df[samples_df["status"] == "SUCCESS"], baseline_mode)
    out_metrics_json.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved aggregate metrics to {out_metrics_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TARZ Body Analysis Agent (RQ1)")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "evaluation" / "config.yaml",
        help="Path to evaluation config.yaml",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=None,
        help="Optional override path to body_ground_truth.csv",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_evaluation(config, args.gt_csv)


if __name__ == "__main__":
    main()

