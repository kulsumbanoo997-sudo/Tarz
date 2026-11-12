"""Smart Catalog Recommendation Engine Evaluation (RQ3) for TARZ IEEE Research Paper.

Evaluates the multi-criteria scoring function S(p) = sum_j w_j * s_j(p):
- Normalized Discounted Cumulative Gain (NDCG@5) against fashion expert ground truth.
- Mean Reciprocal Rank (MRR) for top recommended item.
- Baselines: Size-only ranking and Random ranking (fixed seed, averaged over N trials).
- Inter-stylist agreement via Kendall's tau rank correlation.

Strict Rule: If data is missing or only contains EXAMPLE_ROW, leaves result as TBD and exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("eval_ranking")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "backend"))


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dcg_at_k(relevances: List[float], k: int = 5) -> float:
    """Computes Discounted Cumulative Gain at rank k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = np.array(relevances, dtype=float)
    discounts = np.log2(np.arange(len(gains)) + 2)
    return float(np.sum((2.0**gains - 1.0) / discounts))


def ndcg_at_k(predicted_order_relevances: List[float], ideal_relevances: List[float], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k."""
    dcg = dcg_at_k(predicted_order_relevances, k)
    sorted_ideal = sorted(ideal_relevances, reverse=True)
    idcg = dcg_at_k(sorted_ideal, k)
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def mrr_score(predicted_items: List[str], ground_truth_best: str) -> float:
    """Computes Reciprocal Rank of the single best ground truth item."""
    for rank_idx, item_id in enumerate(predicted_items, start=1):
        if item_id == ground_truth_best:
            return 1.0 / rank_idx
    return 0.0


def rank_items_tarz(profile: dict, catalog_items: List[dict], weights: Optional[dict] = None) -> List[dict]:
    """Ranks products using TARZ scoring engine."""
    from app import scoring

    return scoring.rank(profile, catalog_items, prefs={}, filters={"gender": profile.get("gender", "Women")}, weights=weights)


def rank_items_size_only(profile: dict, catalog_items: List[dict]) -> List[dict]:
    """Size-only baseline: w_size = 1.0, all other weights = 0."""
    size_weights = {"size": 1.0, "shape": 0.0, "tone": 0.0, "style": 0.0, "occasion": 0.0}
    return rank_items_tarz(profile, catalog_items, weights=size_weights)


def evaluate_ranking(config: dict, gt_csv_override: Optional[Path] = None) -> None:
    gt_path = gt_csv_override or (WORKSPACE_ROOT / config["paths"]["ranking_gt_csv"])
    results_dir = WORKSPACE_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    out_metrics_json = results_dir / "ranking_eval_metrics.json"
    out_samples_csv = results_dir / "ranking_eval_samples.csv"

    if not gt_path.is_file():
        logger.warning(f"Ranking ground truth file not found: {gt_path}. Emitting TBD result.")
        result_tbd = {"status": "TBD", "reason": f"File {gt_path} not found"}
        out_metrics_json.write_text(json.dumps(result_tbd, indent=2))
        return

    df = pd.read_csv(gt_path)
    filtered = df[~df["profile_id"].astype(str).str.startswith("EXAMPLE_ROW")].copy()

    if filtered.empty:
        logger.warning(f"No real ranking rows in {gt_path} (only EXAMPLE_ROW rows present).")
        logger.info("Per strict instructions, writing status 'TBD' into Table IV results.")
        result_tbd = {
            "status": "TBD",
            "reason": "Only template EXAMPLE_ROW entries present. Awaiting human stylist rank annotations.",
            "metrics": {
                "ndcg_at_5": {
                    "tarz": "TBD",
                    "baseline_size_only": "TBD",
                    "baseline_random": "TBD",
                },
                "mrr": {
                    "tarz": "TBD",
                    "baseline_size_only": "TBD",
                    "baseline_random": "TBD",
                },
                "inter_stylist_kendall_tau": "TBD",
            },
        }
        out_metrics_json.write_text(json.dumps(result_tbd, indent=2))
        return

    from app import catalog, db

    catalog_items = catalog.load()
    k_eval = config.get("rq3_ranking", {}).get("k", 5)
    random_trials = config.get("rq3_ranking", {}).get("random_baseline_trials", 100)
    seed = config.get("reproducibility", {}).get("seed", 42)
    rng = np.random.default_rng(seed)

    # Inter-stylist agreement (Kendall's tau)
    tau_pvalues = []
    tau_corrs = []
    profile_groups = filtered.groupby("profile_id")

    for pid, group in profile_groups:
        if len(group) >= 2 and group["stylist1_rank"].nunique() > 1 and group["stylist2_rank"].nunique() > 1:
            res = stats.kendalltau(group["stylist1_rank"], group["stylist2_rank"])
            if not np.isnan(res.statistic):
                tau_corrs.append(float(res.statistic))
                tau_pvalues.append(float(res.pvalue))

    mean_tau = float(np.mean(tau_corrs)) if tau_corrs else None

    # Evaluate NDCG and MRR per profile
    per_profile_metrics = []
    tarz_ndcg_list, size_ndcg_list, rand_ndcg_list = [], [], []
    tarz_mrr_list, size_mrr_list, rand_mrr_list = [], [], []

    for pid, group in profile_groups:
        # Load user profile from DB or construct minimal profile
        prof, _ = db.get_profile(str(pid))
        if not prof:
            # Fallback to default Female profile if testing offline profile ID
            prof = {"gender": "Women", "body_shape": "Hourglass", "skin": {"undertone": "warm"}, "measurements": {"bust": 92, "waist": 74, "hip": 98}}

        # Map stylist ranks to relevance scores: rel = max_rank - rank + 1
        max_rank = max(group["stylist1_rank"].max(), group["stylist2_rank"].max(), 5)
        consensus_rank = (group["stylist1_rank"] + group["stylist2_rank"]) / 2.0
        # Highest consensus rank has highest relevance
        relevance_map = {row["garment_id"]: float(max_rank - consensus_rank.loc[idx] + 1) for idx, row in group.iterrows()}
        ideal_best_item = group.loc[consensus_rank.idxmin()]["garment_id"]

        evaluated_ids = set(group["garment_id"])
        eval_items = [p for p in catalog_items if p["id"] in evaluated_ids]
        if len(eval_items) < 2:
            continue

        # TARZ Ranking
        tarz_ranked = rank_items_tarz(prof, eval_items)
        tarz_order = [p["id"] for p in tarz_ranked]
        tarz_rels = [relevance_map.get(i, 0.0) for i in tarz_order]
        all_rels = list(relevance_map.values())

        t_ndcg = ndcg_at_k(tarz_rels, all_rels, k_eval)
        t_mrr = mrr_score(tarz_order, ideal_best_item)

        # Size-only Baseline Ranking
        size_ranked = rank_items_size_only(prof, eval_items)
        size_order = [p["id"] for p in size_ranked]
        size_rels = [relevance_map.get(i, 0.0) for i in size_order]
        s_ndcg = ndcg_at_k(size_rels, all_rels, k_eval)
        s_mrr = mrr_score(size_order, ideal_best_item)

        # Random Ranking Baseline (averaged over 100 trials)
        trial_ndcgs = []
        trial_mrrs = []
        item_ids = list(evaluated_ids)
        for _ in range(random_trials):
            shuffled = list(rng.permutation(item_ids))
            r_rels = [relevance_map.get(i, 0.0) for i in shuffled]
            trial_ndcgs.append(ndcg_at_k(r_rels, all_rels, k_eval))
            trial_mrrs.append(mrr_score(shuffled, ideal_best_item))

        r_ndcg = float(np.mean(trial_ndcgs))
        r_mrr = float(np.mean(trial_mrrs))

        tarz_ndcg_list.append(t_ndcg)
        size_ndcg_list.append(s_ndcg)
        rand_ndcg_list.append(r_ndcg)
        tarz_mrr_list.append(t_mrr)
        size_mrr_list.append(s_mrr)
        rand_mrr_list.append(r_mrr)

        per_profile_metrics.append(
            {
                "profile_id": str(pid),
                "items_evaluated": len(eval_items),
                "tarz_ndcg": round(t_ndcg, 4),
                "size_only_ndcg": round(s_ndcg, 4),
                "random_ndcg": round(r_ndcg, 4),
                "tarz_mrr": round(t_mrr, 4),
                "size_only_mrr": round(s_mrr, 4),
                "random_mrr": round(r_mrr, 4),
            }
        )

    samples_df = pd.DataFrame(per_profile_metrics)
    samples_df.to_csv(out_samples_csv, index=False)
    logger.info(f"Saved per-profile ranking evaluation to {out_samples_csv}")

    final_metrics = {
        "status": "COMPLETED",
        "profiles_evaluated": len(per_profile_metrics),
        "k": k_eval,
        "metrics": {
            "ndcg_at_5": {
                "tarz": round(float(np.mean(tarz_ndcg_list)), 4) if tarz_ndcg_list else "TBD",
                "baseline_size_only": round(float(np.mean(size_ndcg_list)), 4) if size_ndcg_list else "TBD",
                "baseline_random": round(float(np.mean(rand_ndcg_list)), 4) if rand_ndcg_list else "TBD",
            },
            "mrr": {
                "tarz": round(float(np.mean(tarz_mrr_list)), 4) if tarz_mrr_list else "TBD",
                "baseline_size_only": round(float(np.mean(size_mrr_list)), 4) if size_mrr_list else "TBD",
                "baseline_random": round(float(np.mean(rand_mrr_list)), 4) if rand_mrr_list else "TBD",
            },
            "inter_stylist_kendall_tau": round(mean_tau, 4) if mean_tau is not None else "TBD",
        },
    }
    out_metrics_json.write_text(json.dumps(final_metrics, indent=2))
    logger.info(f"Saved aggregate ranking metrics to {out_metrics_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TARZ Smart Catalog Ranking (RQ3)")
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
        help="Optional override path to ranking_gt.csv",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_ranking(config, args.gt_csv)


if __name__ == "__main__":
    main()

