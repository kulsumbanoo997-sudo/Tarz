# TARZ Empirical Evaluation Suite (IEEE Research Paper)

This folder contains the complete, reproducible evaluation pipeline supporting the empirical results in:
**"TARZ: A Multi-Modal Neuro-Symbolic Decision Support and Virtual Try-On Framework for Fashion E-Commerce"**

---

## 1. Directory Structure

```
evaluation/
│
├── config.yaml                     # Central evaluation hyperparameter configuration
├── requirements.txt                # Pinned evaluation dependencies
├── DATA_COLLECTION_GUIDE.md        # SOP for human measurements, photography & stylists
├── README.md                       # This execution manual
│
├── data/                           # Ground truth benchmark templates & CSVs
│   ├── body_ground_truth.csv       # RQ1: Volunteer tape measurements & photo paths
│   ├── tryon_pairs.csv             # RQ2: VTON pairs (raw vs TARZ) & artifact labels
│   ├── ranking_gt.csv              # RQ3: Dual-stylist consensus ranking annotations
│   ├── stylist_testset.csv         # RQ4: Grounded vs unconstrained LLM test cases
│   └── photos/                     # Local volunteer photos (gitignored for privacy)
│
├── results/                        # Generated per-sample outputs & Table IV
│   ├── table4.csv                  # Paper Table IV in CSV format
│   ├── table4.md                   # Paper Table IV in GitHub Markdown
│   ├── body_eval_metrics.json      # RQ1 aggregate metrics & breakdowns
│   ├── body_eval_samples.csv       # RQ1 per-volunteer predictions & errors
│   ├── tryon_eval_metrics.json     # RQ2 aggregate FID, LPIPS, SSIM, artifacts
│   ├── tryon_eval_samples.csv      # RQ2 per-pair perceptual metrics
│   ├── ranking_eval_metrics.json   # RQ3 NDCG@5, MRR & Kendall's tau
│   ├── ranking_eval_samples.csv    # RQ3 per-profile ranking metrics
│   ├── stylist_eval_metrics.json   # RQ4 hallucination & faithfulness metrics
│   ├── stylist_eval_samples.csv    # RQ4 per-case stylist comparison
│   ├── ablation_metrics.json       # Modularity & sensitivity benchmarks
│   └── raw/                        # Full raw LLM traces (gitignored)
│
└── tests/                          # Unit tests for evaluation metric formulas
    └── test_metrics.py             # Hand-computed deterministic metric verifications
```

---

## 2. Scientific Reporting Integrity Rule (Non-Fabrication Policy)

> [!IMPORTANT]
> **Strict Non-Fabrication Protocol:**
> In compliance with IEEE research guidelines, no benchmark results are simulated, hard-coded, or fabricated. 
> - Template rows beginning with `EXAMPLE_ROW` in the CSV files are dummy examples and are **automatically skipped**.
> - If a ground-truth dataset has not yet been populated with real human data, the evaluation scripts log a notice and strictly output **`"TBD"`** for the corresponding Table IV cells.
> - Synthetic test data is strictly isolated to `evaluation/tests/` and never written to `evaluation/results/`.

---

## 3. Setup & Installation

Install evaluation dependencies into your Python 3.12 environment:

```bash
pip install -r evaluation/requirements.txt
```

Verify that all evaluation metric calculations are working:

```bash
python -m pytest -q evaluation/tests
```

---

## 4. End-to-End Evaluation Workflow

Follow this exact order to execute the evaluation pipeline:

### Step 1: Collect & Populate Ground Truth Data
Consult [`DATA_COLLECTION_GUIDE.md`](DATA_COLLECTION_GUIDE.md) for target sample sizes and protocols.
Fill in the CSV files in `evaluation/data/`:
* `body_ground_truth.csv`: Tape measurements and photo paths for $\ge 20$ participants.
* `tryon_pairs.csv`: Paired outputs and three-arm artifact labels for $\ge 30$ pairs.
* `ranking_gt.csv`: Dual fashion expert rank annotations for $\ge 15$ profiles.
* `stylist_testset.csv`: Evaluation cases for $\ge 30$ styling recommendations.

### Step 2: Evaluate Body Analysis Agent (RQ1)
Evaluates circumference MAE (bust, hip), body shape classification accuracy, skin depth/undertone accuracy, and population-mean baseline:

```bash
python evaluation/eval_body.py
```
*Outputs: `results/body_eval_metrics.json`, `results/body_eval_samples.csv`.*

### Step 3: Evaluate Smart Catalog Recommendation Ranking (RQ3)
Computes NDCG@5, MRR, size-only baseline, random ranking baseline (100 trials), and inter-stylist Kendall's $\tau$:

```bash
python evaluation/eval_ranking.py
```
*Outputs: `results/ranking_eval_metrics.json`, `results/ranking_eval_samples.csv`.*

### Step 4: Evaluate Honest Stylist Agent (RQ4)
Compares the unconstrained standard LLM vs TARZ Honest Stylist on hallucination rate, faithfulness score, and template fallback share:

```bash
python evaluation/eval_stylist.py
```
*Outputs: `results/stylist_eval_metrics.json`, `results/stylist_eval_samples.csv`, `results/raw/case_*.json`.*

### Step 5: Evaluate Virtual Try-On Pipeline (RQ2)
Computes FID, LPIPS perceptual similarity, unmasked SSIM, and three-arm artifact rates (Raw baseline vs TARZ extraction):

```bash
python evaluation/eval_tryon.py
```
*(Optional: add `--batch-run` to execute active generation for pairs).*  
*Outputs: `results/tryon_eval_metrics.json`, `results/tryon_eval_samples.csv`.*

### Step 6: Execute Component Ablation & Sensitivity Analysis
Runs score term leave-one-out ablation, $\pm 20\%$ weight perturbation analysis, decision boundary threshold sensitivity, and failover tier benchmarks:

```bash
python evaluation/eval_ablation.py
```
*Outputs: `results/ablation_metrics.json`.*

### Step 7: Compile Paper Table IV
Consolidates all metrics from `results/*.json` into publication-ready formats:

```bash
python evaluation/make_table4.py
```
*Outputs: `results/table4.csv`, `results/table4.md`.*

