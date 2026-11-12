"""Unit Tests for TARZ Evaluation Metric Functions.

Uses deterministic, hand-computed synthetic fixtures.
Strict Rule: These tests live in evaluation/tests/ and never write to evaluation/results/.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from evaluation.eval_ranking import dcg_at_k, ndcg_at_k, mrr_score
from evaluation.eval_stylist import evaluate_hallucinations, evaluate_faithfulness
from evaluation.eval_body import calculate_baseline_predictions
from evaluation.make_table4 import bootstrap_ci


def test_dcg_and_ndcg_hand_computed():
    # Relevances [3, 2, 1]
    # DCG = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) + (2^1 - 1)/log2(4)
    # DCG = 7/1 + 3/1.5849625 + 1/2 = 7 + 1.892789 + 0.5 = 9.392789
    ideal = [3.0, 2.0, 1.0]
    expected_idcg = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    calc_dcg = dcg_at_k(ideal, k=3)
    assert abs(calc_dcg - expected_idcg) < 1e-4

    # Perfect ranking should have NDCG = 1.0
    assert abs(ndcg_at_k(ideal, ideal, k=3) - 1.0) < 1e-5

    # Reverse ranking: [1, 2, 3]
    rev = [1.0, 2.0, 3.0]
    rev_dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**3 - 1) / math.log2(4)
    expected_ndcg = rev_dcg / expected_idcg
    assert abs(ndcg_at_k(rev, ideal, k=3) - expected_ndcg) < 1e-4


def test_mrr_hand_computed():
    preds = ["p02", "p06", "p01", "p08"]
    # Target item is at rank 3 -> MRR = 1/3
    assert abs(mrr_score(preds, "p01") - (1.0 / 3.0)) < 1e-6
    # Target item is at rank 1 -> MRR = 1.0
    assert mrr_score(preds, "p02") == 1.0
    # Target item not in predictions -> MRR = 0.0
    assert mrr_score(preds, "p99") == 0.0


def test_baseline_body_predictions():
    preds_women = calculate_baseline_predictions(165.0, "Women", baseline_mode="population_mean")
    assert "bust" in preds_women and "hip" in preds_women and "waist" in preds_women
    # Size M for women is (88, 94) bust -> midpoint 91.0, (94, 100) hip -> midpoint 97.0
    assert preds_women["bust"] == 91.0
    assert preds_women["hip"] == 97.0


def test_hallucination_detection():
    evidence = {"verdict": "suitable"}
    valid_candidates = {"p01", "p02", "p03"}

    # 1. Honest, faithful response
    honest_output = {
        "verdict": "suitable",
        "headline": "Great fit",
        "reasons": ["Tone (1): matches tone", "Size (0.9): fits well"],
        "alternatives": [{"id": "p01", "why": "Good color"}],
    }
    res_honest = evaluate_hallucinations(honest_output, evidence, valid_candidates)
    assert res_honest["has_hallucination"] is False
    assert res_honest["verdict_match"] is True

    # 2. Hallucinated response: verdict flip
    flipped_output = {
        "verdict": "unsuitable",
        "alternatives": [{"id": "p01", "why": "alt"}],
    }
    res_flip = evaluate_hallucinations(flipped_output, evidence, valid_candidates)
    assert res_flip["has_hallucination"] is True
    assert res_flip["verdict_match"] is False

    # 3. Hallucinated response: invented candidate ID
    invented_output = {
        "verdict": "suitable",
        "alternatives": [{"id": "p99_NON_EXISTENT", "why": "fake"}],
    }
    res_inv = evaluate_hallucinations(invented_output, evidence, valid_candidates)
    assert res_inv["has_hallucination"] is True
    assert "p99_NON_EXISTENT" in res_inv["invalid_ids"]


def test_faithfulness_scoring():
    evidence = {}
    # 2 out of 3 reasons cite verified score terms (tone, shape)
    output = {
        "reasons": [
            "Tone (1): in your suggested colour family",
            "Shape (1): suits your body shape well",
            "This garment looks absolutely stunning and fashionable!",
        ]
    }
    faith_score = evaluate_faithfulness(output, evidence)
    assert abs(faith_score - (2.0 / 3.0)) < 1e-4

    # 0 reasons cited
    assert evaluate_faithfulness({"reasons": []}, evidence) == 0.0


def test_bootstrap_ci():
    values = [10.0, 11.0, 12.0, 10.5, 11.5]
    mean, std, ci_low, ci_high = bootstrap_ci(values, n_boot=500, ci=0.95, seed=42)
    assert 10.0 <= mean <= 12.0
    assert ci_low <= mean <= ci_high
    assert std > 0

