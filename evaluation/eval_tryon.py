"""Virtual Try-On (VTON) Pipeline Evaluation (RQ2) for TARZ IEEE Research Paper.

Evaluates generative quality and anatomical consistency:
- Fréchet Inception Distance (FID): Baseline raw-image vs TARZ extraction.
- Learned Perceptual Image Patch Similarity (LPIPS): Baseline vs TARZ.
- Structural Similarity Index (SSIM) on unmasked background & skin preservation.
- Three-Arm Artifact Rate (%): Human-labeled anatomical distortion rate.
- Optional batch execution mode (--batch-run) testing latency and failover tiers.

Strict Rule: If data is missing or only contains EXAMPLE_ROW, leaves result as TBD and exits cleanly.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("eval_tryon")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_unmasked_ssim(
    img_orig_path: Path,
    img_gen_path: Path,
    mask_path: Optional[Path] = None,
) -> Optional[float]:
    """Computes SSIM strictly over the unmasked regions (face, hair, lower body, background)."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        logger.warning("scikit-image not installed. Run: pip install scikit-image")
        return None

    if not img_orig_path.is_file() or not img_gen_path.is_file():
        return None

    im_orig = Image.open(img_orig_path).convert("RGB")
    im_gen = Image.open(img_gen_path).convert("RGB").resize(im_orig.size, Image.Resampling.LANCZOS)

    arr_orig = np.array(im_orig)
    arr_gen = np.array(im_gen)

    # Determine mask (0 = evaluate/unmasked, 255 = clothing/masked)
    if mask_path and mask_path.is_file():
        m_img = Image.open(mask_path).convert("L").resize(im_orig.size, Image.Resampling.NEAREST)
        m_arr = np.array(m_img)
        unmasked = m_arr < 128
    else:
        # Default heuristic: Torso region (height 25% to 75%, width 20% to 80%) is considered clothing
        h, w = arr_orig.shape[:2]
        unmasked = np.ones((h, w), dtype=bool)
        unmasked[int(0.25 * h) : int(0.75 * h), int(0.20 * w) : int(0.80 * w)] = False

    if unmasked.sum() == 0:
        return None

    # Compute grayscale SSIM
    gray_orig = np.array(im_orig.convert("L"))
    gray_gen = np.array(im_gen.convert("L"))

    score, diff = ssim(gray_orig, gray_gen, full=True)
    unmasked_score = float(diff[unmasked].mean())
    return round(unmasked_score, 4)


def calculate_lpips(
    img_a_path: Path,
    img_b_path: Path,
    lpips_fn: Any = None,
    device: str = "cpu",
) -> Optional[float]:
    """Computes Learned Perceptual Image Patch Similarity using AlexNet."""
    try:
        import torch
        import torchvision.transforms as transforms
    except ImportError:
        return None

    if lpips_fn is None:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(device)
        except Exception:
            return None

    if not img_a_path.is_file() or not img_b_path.is_file():
        return None

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    try:
        t_a = transform(Image.open(img_a_path).convert("RGB")).unsqueeze(0).to(device)
        t_b = transform(Image.open(img_b_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            dist = float(lpips_fn(t_a, t_b).item())
        return round(dist, 4)
    except Exception as e:
        logger.warning(f"LPIPS computation failed: {e}")
        return None


def calculate_fid_directories(
    dir_real: Path,
    dir_fake: Path,
    device: str = "cpu",
) -> Optional[float]:
    """Computes Fréchet Inception Distance between two directories."""
    try:
        from cleanfid import fid
        score = fid.compute_fid(str(dir_real), str(dir_fake), mode="clean", device=device, num_workers=0)
        return round(float(score), 2)
    except Exception as e:
        logger.warning(f"clean-fid computation unavailable or failed: {e}")
        return None


async def run_batch_tryon(filtered_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    """Optionally runs try-on generation for baseline vs TARZ extraction."""
    from app import catalog, config, tryon

    batch_out_dir = results_dir / "generated_tryons"
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    updated_rows = []
    logger.info(f"Running batch VTON generation for {len(filtered_df)} pairs...")

    for idx, row in filtered_df.iterrows():
        pair_id = str(row["pair_id"])
        garment_id = str(row["garment_id"])
        person_photo = WORKSPACE_ROOT / str(row.get("person_photo_path", ""))

        if not person_photo.is_file():
            logger.warning(f"Person photo for {pair_id} not found: {person_photo}")
            continue

        product = catalog.get(garment_id)
        if not product:
            continue

        # 1. Run TARZ (with Smart Crop)
        t0 = time.time()
        tarz_res = await tryon.run_tryon(str(person_photo), product)
        t_tarz = time.time() - t0

        row["tarz_output_path"] = tarz_res["path"]
        row["tarz_backend"] = tarz_res["backend"]
        row["tarz_latency_s"] = round(t_tarz, 2)

        # 2. Run Baseline (Raw Garment image without MediaPipe smart crop)
        # letterbox without category-aware cropping
        t0 = time.time()
        # simulate raw letterbox
        raw_garment_path = config.UPLOAD_DIR / f"raw_garment_{pair_id}.jpg"
        async with httpx.AsyncClient(timeout=20) as cx:
            r = await cx.get(product["image"])
            raw_garment_path.write_bytes(r.content)

        # Execute IDM-VTON with raw garment
        raw_res = await tryon.run_tryon(str(person_photo), {**product, "image": str(raw_garment_path)})
        t_raw = time.time() - t0

        row["raw_output_path"] = raw_res["path"]
        row["raw_backend"] = raw_res["backend"]
        row["raw_latency_s"] = round(t_raw, 2)

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)


def evaluate_tryon(config: dict, pairs_csv_override: Optional[Path] = None, batch_run: bool = False) -> None:
    pairs_path = pairs_csv_override or (WORKSPACE_ROOT / config["paths"]["tryon_pairs_csv"])
    results_dir = WORKSPACE_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out_metrics_json = results_dir / "tryon_eval_metrics.json"
    out_samples_csv = results_dir / "tryon_eval_samples.csv"

    if not pairs_path.is_file():
        logger.warning(f"Try-on pairs file not found: {pairs_path}. Emitting TBD result.")
        result_tbd = {"status": "TBD", "reason": f"File {pairs_path} not found"}
        out_metrics_json.write_text(json.dumps(result_tbd, indent=2))
        return

    df = pd.read_csv(pairs_path)
    filtered = df[~df["pair_id"].astype(str).str.startswith("EXAMPLE_ROW")].copy()

    if filtered.empty:
        logger.warning(f"No real tryon pairs in {pairs_path} (only EXAMPLE_ROW rows present).")
        logger.info("Per strict instructions, writing status 'TBD' into Table IV results.")
        result_tbd = {
            "status": "TBD",
            "reason": "Only template EXAMPLE_ROW entries present. Awaiting human try-on paired samples.",
            "metrics": {
                "fid": {
                    "tarz": "TBD",
                    "baseline_raw": "TBD",
                },
                "lpips": {
                    "tarz": "TBD",
                    "baseline_raw": "TBD",
                },
                "ssim_unmasked": {
                    "tarz": "TBD",
                    "baseline_raw": "TBD",
                },
                "three_arm_artifact_rate_pct": {
                    "tarz": "TBD",
                    "baseline_raw": "TBD",
                },
            },
        }
        out_metrics_json.write_text(json.dumps(result_tbd, indent=2))
        return

    # Check for optional batch generation
    if batch_run:
        import asyncio
        filtered = asyncio.run(run_batch_tryon(filtered, results_dir))
        filtered.to_csv(pairs_path, index=False)

    # Initialize LPIPS if torch is present
    device = config.get("rq2_tryon", {}).get("compute_device", "cpu")
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net=config.get("rq2_tryon", {}).get("lpips_net", "alex")).to(device)
    except Exception:
        logger.info("LPIPS package not loaded. Skipping perceptual LPIPS computation if unavailable.")

    # Evaluate per-sample pairs
    eval_records = []
    tarz_lpips_list, raw_lpips_list = [], []
    tarz_ssim_list, raw_ssim_list = [], []

    for idx, row in filtered.iterrows():
        pair_id = str(row["pair_id"])
        raw_path = WORKSPACE_ROOT / str(row["raw_output_path"])
        tarz_path = WORKSPACE_ROOT / str(row["tarz_output_path"])
        mask_path = WORKSPACE_ROOT / str(row["person_mask_path"]) if pd.notnull(row.get("person_mask_path")) else None

        # Artifact labels (0 or 1)
        raw_art = int(row["three_arm_artifact_raw"]) if pd.notnull(row.get("three_arm_artifact_raw")) else None
        tarz_art = int(row["three_arm_artifact_tarz"]) if pd.notnull(row.get("three_arm_artifact_tarz")) else None

        # Reference photo (original person photo) for fidelity to unmasked regions
        person_photo = WORKSPACE_ROOT / str(row["person_photo_path"]) if pd.notnull(row.get("person_photo_path")) else None
        ref_path = person_photo if (person_photo and person_photo.is_file()) else raw_path

        # SSIM unmasked
        ssim_raw = calculate_unmasked_ssim(ref_path, raw_path, mask_path)
        ssim_tarz = calculate_unmasked_ssim(ref_path, tarz_path, mask_path)

        # LPIPS perceptual similarity
        lp_raw = calculate_lpips(ref_path, raw_path, lpips_fn, device)
        lp_tarz = calculate_lpips(ref_path, tarz_path, lpips_fn, device)

        if ssim_tarz is not None:
            tarz_ssim_list.append(ssim_tarz)
        if ssim_raw is not None:
            raw_ssim_list.append(ssim_raw)
        if lp_tarz is not None:
            tarz_lpips_list.append(lp_tarz)
        if lp_raw is not None:
            raw_lpips_list.append(lp_raw)

        eval_records.append(
            {
                "pair_id": pair_id,
                "garment_id": row.get("garment_id"),
                "arms_pose": row.get("arms_pose", "relaxed"),
                "raw_artifact": raw_art,
                "tarz_artifact": tarz_art,
                "raw_ssim_unmasked": ssim_raw,
                "tarz_ssim_unmasked": ssim_tarz,
                "raw_lpips": lp_raw,
                "tarz_lpips": lp_tarz,
            }
        )

    samples_df = pd.DataFrame(eval_records)
    samples_df.to_csv(out_samples_csv, index=False)
    logger.info(f"Saved per-pair tryon evaluation to {out_samples_csv}")

    # Artifact rate %
    raw_arts = samples_df["raw_artifact"].dropna()
    tarz_arts = samples_df["tarz_artifact"].dropna()

    raw_art_rate = float(raw_arts.mean() * 100.0) if len(raw_arts) else None
    tarz_art_rate = float(tarz_arts.mean() * 100.0) if len(tarz_arts) else None

    # Compute FID across output folders if enough images exist (>= 10 images)
    tarz_dirs = [tarz_path.parent for idx, row in filtered.iterrows() if (WORKSPACE_ROOT / str(row["tarz_output_path"])).is_file()]
    raw_dirs = [raw_path.parent for idx, row in filtered.iterrows() if (WORKSPACE_ROOT / str(row["raw_output_path"])).is_file()]

    fid_tarz = None
    fid_raw = None
    ref_photo_dir = WORKSPACE_ROOT / "evaluation" / "data" / "photos"
    if len(tarz_dirs) >= 10 and len(raw_dirs) >= 10:
        if ref_photo_dir.is_dir() and len(list(ref_photo_dir.glob("*.jpg"))) >= 10:
            fid_raw = calculate_fid_directories(ref_photo_dir, raw_dirs[0], device)
            fid_tarz = calculate_fid_directories(ref_photo_dir, tarz_dirs[0], device)
        else:
            fid_tarz = calculate_fid_directories(raw_dirs[0], tarz_dirs[0], device)

    final_metrics = {
        "status": "COMPLETED",
        "sample_count": len(samples_df),
        "metrics": {
            "fid": {
                "tarz": round(fid_tarz, 2) if fid_tarz is not None else "TBD",
                "baseline_raw": round(fid_raw, 2) if fid_raw is not None else "TBD",
            },
            "lpips": {
                "tarz": round(float(np.mean(tarz_lpips_list)), 4) if tarz_lpips_list else "TBD",
                "baseline_raw": round(float(np.mean(raw_lpips_list)), 4) if raw_lpips_list else "TBD",
            },
            "ssim_unmasked": {
                "tarz": round(float(np.mean(tarz_ssim_list)), 4) if tarz_ssim_list else "TBD",
                "baseline_raw": round(float(np.mean(raw_ssim_list)), 4) if raw_ssim_list else "TBD",
            },
            "three_arm_artifact_rate_pct": {
                "tarz": round(tarz_art_rate, 2) if tarz_art_rate is not None else "TBD",
                "baseline_raw": round(raw_art_rate, 2) if raw_art_rate is not None else "TBD",
            },
        },
    }
    out_metrics_json.write_text(json.dumps(final_metrics, indent=2))
    logger.info(f"Saved aggregate tryon metrics to {out_metrics_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TARZ Virtual Try-On Pipeline (RQ2)")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "evaluation" / "config.yaml",
        help="Path to evaluation config.yaml",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="Optional override path to tryon_pairs.csv",
    )
    parser.add_argument(
        "--batch-run",
        action="store_true",
        help="Execute active batch inference for both raw and TARZ extraction backends",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_tryon(config, args.pairs_csv, args.batch_run)


if __name__ == "__main__":
    main()

