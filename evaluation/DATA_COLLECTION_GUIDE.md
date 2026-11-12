# Ground Truth Data Collection Protocol for TARZ Evaluation

This guide outlines the standard operating procedure (SOP) for collecting empirical benchmark data for the research paper:
**"TARZ: A Multi-Modal Neuro-Symbolic Decision Support and Virtual Try-On Framework for Fashion E-Commerce"**

---

## 1. Target Sample Sizes (Minimum Thresholds for Statistical Significance)

To achieve statistically significant comparisons with 95% bootstrap confidence intervals for IEEE paper publication, collect at least:

| Dataset / Component | Target Sample Size | Description |
|---|---|---|
| **Body Analysis (RQ1)** | $\ge 20$ human volunteers | Diverse heights, genders (Women/Men), skin tones, and body shapes |
| **Virtual Try-On (RQ2)** | $\ge 30$ try-on pairs | Pairing volunteers with catalog items across categories (Kurtis, Tops, Dresses) |
| **Recommendation Ranking (RQ3)** | $\ge 15$ profiles $\times 2$ stylists | 15 distinct profiles evaluated and ranked by 2 independent fashion experts |
| **Honest Stylist (RQ4)** | $\ge 30$ test cases | Including balanced mix of suitable, partly suitable, and deliberately unsuitable items |

---

## 2. Standardized Photography Protocol

Single-camera 2D anthropometric estimation requires high visual consistency:

1. **Camera Framing & Distance:**
   - Mount smartphone or camera at approximately **waist height** ($\sim 90\text{--}100\text{ cm}$ from floor), perfectly level (no downward/upward tilt).
   - Distance: 2.5 to 3.0 meters from subject.
   - Framing: **Full body strictly required** — top of head to bare feet/ankles must be clearly visible inside the frame.
2. **Subject Posture:**
   - Stand straight, facing directly into the camera lens with feet hip-width apart.
   - **Arms Pose:** 
     - Primary evaluation set: **Arms relaxed naturally at the sides** (gap between arms and torso).
     - Controlled ablation set: **Arms crossed / folded over chest** (used specifically to evaluate RQ2 three-arm artifact robustness).
3. **Clothing:**
   - Form-fitting or light clothing (e.g., leggings, fitted t-shirt, tank top).
   - Avoid bulky jackets, oversized hoodies, or long flared skirts that obscure natural anatomical hip/shoulder contours.
4. **Lighting & Background:**
   - Plain, solid light-colored wall (white, off-white, light grey).
   - Even, diffused lighting. Avoid harsh shadows or strong backlighting from windows.

---

## 3. Physical Anthropometric Tape Measurement SOP

All ground truth measurements must be recorded in **centimeters (cm)** using a non-elastic tailor's tape:

* **Stature (Height):** Measured barefoot against a vertical wall from crown to floor.
* **Bust / Chest:** Measured horizontally around the fullest part of the chest/bust, keeping the tape parallel to the floor under the armpits.
* **Waist:** Measured horizontally at the natural waistline (narrowest part of the torso, typically 2–3 cm above the navel).
* **Hip:** Measured horizontally around the widest part of the buttocks/pelvis.
* **Skin Depth & Undertone Ground Truth:**
  - Standard Fitzpatrick skin type scale (I–VI) mapped to: `Fair` (I–II), `Medium` (III–IV), `Deep` (V–VI).
  - Undertone determined by standard wrist vein / white cloth test: `warm` (greenish veins, golden hues), `cool` (bluish veins, pink/rosy hues), or `neutral`.

---

## 4. Subject Consent & Privacy Protocol

All human evaluations must adhere to ethical data collection principles:

1. **Informed Consent:** Every volunteer must confirm consent (`consent_given = yes`) before participation.
2. **Storage Anonymity:** Participant identifiers must be anonymized (`p_01`, `p_02`, etc.). Never record real names, emails, or phone numbers in CSV files.
3. **Local Storage Only:** Raw volunteer photos stored in `evaluation/data/photos/` are strictly excluded from git tracking via `.gitignore`.
4. **Right to Withdrawal:** Any participant may request immediate deletion of their photographic data.

---

## 5. CSV File Formatting Summary

1. `body_ground_truth.csv`: Fill real volunteer measurements and local image paths.
2. `tryon_pairs.csv`: Specify `raw_output_path` (baseline without smart crop) and `tarz_output_path` (with TARZ pose-guided smart crop), and mark `three_arm_artifact` flags ($0$ = normal, $1$ = extra limb visible).
3. `ranking_gt.csv`: Two stylists independently assign rank order ($1, 2, 3, \dots$) to candidates for each profile.
4. `stylist_testset.csv`: Provide test cases with a flag for whether the garment is deliberately unsuitable for that body type/tone to test honest refusal.

> [!NOTE]
> Rows beginning with `EXAMPLE_ROW` are dummy templates and are **automatically skipped** by all TARZ evaluation scripts.

