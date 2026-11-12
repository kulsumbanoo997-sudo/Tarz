# TARZ: Multi-Modal AI Decision-Support and Virtual Try-On System for Fashion E-Commerce

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)](https://vitejs.dev/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-FF6F00.svg)](https://mediapipe.dev/)
[![Diffusion Model](https://img.shields.io/badge/VTON-IDM--VTON%20(SDXL)-8A2BE2.svg)](https://github.com/yisol/IDM-VTON)

> **Academic & Technical Whitepaper Documentation**  
> *A comprehensive architectural reference, mathematical formulation, and implementation manual for research publications.*

---

## 1. Abstract & Executive Summary

Online fashion retail suffers from excessive return rates (exceeding 30–40% globally), predominantly driven by sizing discrepancies, unflattering silhouettes, undertone mismatches, and inaccurate visual expectations. **TARZ** is a multi-modal, neuro-symbolic fashion decision-support and visual demonstration framework.

TARZ integrates four foundational components into an orchestrated end-to-end pipeline:
1. **Computer Vision Body Analysis Agent:** Uses single-camera monocular RGB pose landmark estimation (MediaPipe 33 keypoints) combined with anthropometric projection heuristics to extract body shape ($\rho$), skin tone luminance ($Y$), undertone ($G-B$), and estimated circumferences (bust, waist, hip).
2. **Multi-Criteria Recommendation Engine (Smart Catalog):** Computes an explainable, multi-objective score $S(p) = \sum w_j \cdot s_j(p)$ over fit confidence, body shape compatibility, skin tone harmony, aesthetic preferences, and occasion alignment with configurable silhouette strictness penalties.
3. **High-Fidelity Virtual Try-On (VTON) Pipeline:** Incorporates intelligent garment extraction via pose-guided bounding, aspect-ratio preservation, and Stable Diffusion XL (SDXL)-driven latent diffusion inpainting with GarmentNet and IP-Adapter conditioning (IDM-VTON). Includes an automated failover system (Authenticated ZeroGPU $\to$ Anonymous Fallback $\to$ Cloud API Fallback).
4. **Honest Stylist Agent (Guardrailed Neuro-Symbolic LLM):** Employs strict server-side grounding where the deterministic mathematical engine controls the purchase verdict, while a high-speed LLM (Groq / Gemini) generates explanations bounded strictly to verified evidence records, with fallback to rule-based templating.

```
                           +------------------------+
                           |  User Input (RGB Image |
                           |    or Measurements)    |
                           +-----------+------------+
                                       |
                                       v
                     +-----------------------------------+
                     |   Body Analysis Agent (CV/AI)     |
                     |   MediaPipe Pose + Skin Patches   |
                     |   - Body Shape (rho)              |
                     |   - Skin Undertone (G - B)        |
                     |   - Recommended Size & Fit Conf.  |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |   Smart Catalog Scoring Engine    |
                     |   S(p) = Sum w_j * s_j(p)         |
                     |   Deterministic Purchase Verdict  |
                     +--------+-----------------+--------+
                              |                 |
            +-----------------+                 +------------------+
            |                                                      |
            v                                                      v
+-----------------------+                              +-----------------------+
|  Virtual Try-On (VTON)|                              |  Honest Stylist Agent |
|  - Smart Garment Crop |                              |  - Guardrailed LLM    |
|  - SDXL + IP-Adapter  |                              |  - Candidate Filter   |
|  - IDM-VTON Diffusion |                              |  - Zero Hallucination |
+-----------+-----------+                              +-----------+-----------+
            |                                                      |
            +-------------------------+----------------------------+
                                      |
                                      v
                         +--------------------------+
                         | Explainable Frontend UI  |
                         | React + Vite Client      |
                         +--------------------------+
```

---

## 2. Theoretical Framework & Mathematical Formulations

### 2.1 Anthropometric Body Estimation & Classification

#### A. Stature Scaling & Circumference Approximations
Given a user-provided stature $H$ (in centimeters) and 2D pixel coordinates of detected anatomical landmarks $(x_i, y_i)$, vertical pixel height is calibrated against the nose-to-ankle fraction:
$$\text{Scale Factor } k = \frac{H \cdot \text{CAL}_{\text{nose\_to\_ankle}}}{|\bar{y}_{\text{ankle}} - y_{\text{nose}}|}$$
where $\text{CAL}_{\text{nose\_to\_ankle}} = 0.885$.

Transverse body breadths ($W$) and sagittal depths ($D$) are modeled as ellipses using anatomical constants:
$$W_{\text{chest}} = k \cdot \|P_{\text{L\_shoulder}} - P_{\text{R\_shoulder}}\| \cdot \text{CAL}_{\text{chest\_width}}$$
$$D_{\text{chest}} = W_{\text{chest}} \cdot \text{CAL}_{\text{chest\_depth\_ratio}}$$
$$W_{\text{hip}} = k \cdot \|P_{\text{L\_hip}} - P_{\text{R\_hip}}\| \cdot \text{CAL}_{\text{hip\_body\_factor}}$$
$$D_{\text{hip}} = W_{\text{hip}} \cdot \text{CAL}_{\text{hip\_depth\_ratio}}$$
with $\text{CAL}_{\text{chest\_width}} = 0.90$, $\text{CAL}_{\text{chest\_depth\_ratio}} = 0.66$, $\text{CAL}_{\text{hip\_body\_factor}} = 1.20$, and $\text{CAL}_{\text{hip\_depth\_ratio}} = 0.70$.

Circumferences ($C$) are computed using Ramanujan's second approximation for ellipse perimeter ($a = W/2, b = D/2$):
$$C \approx \pi \left[ 3(a + b) - \sqrt{(3a + b)(a + 3b)} \right]$$

#### B. Morphological Body Shape Index ($\rho$)
The body shape index is defined as the shoulder-to-hip breadth ratio:
$$\rho = \frac{\|P_{\text{L\_shoulder}} - P_{\text{R\_shoulder}}\|}{\|P_{\text{L\_hip}} - P_{\text{R\_hip}}\|}$$

The classification partition $f_{\text{shape}}(\rho)$ is defined by piecewise boundaries:
$$f_{\text{shape}}(\rho) = \begin{cases} 
\text{Inverted Triangle}, & \rho > 1.20 \\
\text{Apple}, & 1.06 < \rho \le 1.20 \\
\text{Hourglass}, & 0.94 \le \rho \le 1.06 \\
\text{Rectangle}, & 0.82 \le \rho < 0.94 \\
\text{Pear}, & \rho < 0.82
\end{cases}$$

For manual measurement mode, waist-to-hip ratio $w$ and differential bust-to-hip percentage $\delta = \frac{\text{bust} - \text{hip}}{\text{hip}}$ are used:
$$f_{\text{manual}}(\text{bust}, \text{waist}, \text{hip}) = \begin{cases}
\text{Inverted Triangle}, & \delta > 0.05 \\
\text{Pear}, & \delta < -0.05 \\
\text{Hourglass}, & |\delta| \le 0.05 \land w \le 0.75 \\
\text{Apple}, & |\delta| \le 0.05 \land w \ge 0.90 \\
\text{Rectangle}, & \text{otherwise}
\end{cases}$$

#### C. Photometric Skin Tone & Undertone Decomposition
Skin patches are sampled bilaterally beneath the zygomatic landmarks (cheeks) using rotation-invariant offsets aligned with inter-pupillary tilt angle $\theta = \text{atan2}(\Delta y_{\text{eyes}}, \Delta x_{\text{eyes}})$.

Color decomposition operates in linear RGB space:
* **Perceived Luminance ($Y$):**
  $$Y = 0.299R + 0.587G + 0.114B$$
  $$\text{Depth} = \begin{cases} \text{Fair}, & Y > 170 \\ \text{Medium}, & 110 < Y \le 170 \\ \text{Deep}, & Y \le 110 \end{cases}$$
* **Chrominance Undertone ($\Delta_{\text{gb}}$):**
  $$\Delta_{\text{gb}} = G - B$$
  $$\text{Undertone} = \begin{cases} \text{Warm}, & \Delta_{\text{gb}} > 5.0 \\ \text{Cool}, & \Delta_{\text{gb}} < -1.0 \\ \text{Neutral}, & -1.0 \le \Delta_{\text{gb}} \le 5.0 \end{cases}$$

---

### 2.2 Sizing Distance & Multi-Criteria Scoring

#### A. Sizing Fit Distance Metric
Let $M = \{\text{bust}, \text{waist}, \text{hip}\}$. For a target size band $S \in \{\text{S}, \text{M}, \text{L}, \text{XL}, \text{XXL}\}$ with bounding intervals $[L_m, H_m]$:
$$d_m(v_m, [L_m, H_m]) = \begin{cases} 
0.0, & L_m \le v_m \le H_m \\
\frac{L_m - v_m}{H_m - L_m}, & v_m < L_m \\
\frac{v_m - H_m}{H_m - L_m}, & v_m > H_m
\end{cases}$$

The overall fit confidence metric $c(S)$ is:
$$c(S) = \max\left(0.0, 1.0 - \frac{1}{|M|} \sum_{m \in M} \min(1.0, d_m)\right)$$

The optimal recommended size $S^*$ minimizes average distance:
$$S^* = \arg\min_{S} \left( \frac{1}{|M|} \sum_{m \in M} d_m(S) \right)$$

#### B. Explainable Multi-Criteria Recommendation Function
The composite match score $S(p)$ for product $p$ is a convex combination:
$$S(p) = \sum_{j \in \{\text{size}, \text{shape}, \text{tone}, \text{style}, \text{occasion}\}} w_j \cdot s_j(p), \quad \text{where } \sum w_j = 1, \ w_j \ge 0$$

Individual sub-scoring terms $s_j(p) \in [0, 1]$ are computed as follows:

1. **Size Fit Term ($s_{\text{size}}$):**
   Incorporates silhouette strictness coefficient $\lambda_{\text{fit}} \in \{1.5 \text{ (fitted/structured)}, 1.0 \text{ (aline)}, 0.6 \text{ (flowy)}\}$:
   $$s_{\text{size}}(p) = \max\left(0.0, 1.0 - \min(1.0, \lambda_{\text{fit}}(p) \cdot (1.0 - c(S^*)))\right)$$
2. **Silhouette / Shape Compatibility ($s_{\text{shape}}$):**
   $$s_{\text{shape}}(p) = \begin{cases} 1.0, & \text{bodyFit}(p) \in \text{PermissibleFits}(\text{body\_shape}) \\ 0.0, & \text{otherwise} \end{cases}$$
3. **Color Harmony ($s_{\text{tone}}$):**
   $$s_{\text{tone}}(p) = \begin{cases} 1.0, & \text{colorFamily}(p) = \text{undertone} \\ 0.5, & \text{undertone is neutral or unknown} \\ 0.0, & \text{colorFamily}(p) \ne \text{undertone} \end{cases}$$
4. **Style Alignment ($s_{\text{style}}$):**
   $$s_{\text{style}}(p) = \begin{cases} 1.0, & \text{styles}_{\text{pref}} \cap \text{styles}(p) \ne \emptyset \\ 0.5, & \text{styles}_{\text{pref}} = \emptyset \\ 0.0, & \text{otherwise} \end{cases}$$
5. **Occasion Match ($s_{\text{occ}}$):**
   $$s_{\text{occ}}(p) = \begin{cases} 1.0, & \text{occasion}_{\text{pref}} \in \text{occasions}(p) \\ 0.5, & \text{occasion}_{\text{pref}} = \emptyset \\ 0.0, & \text{otherwise} \end{cases}$$

#### C. Decision Boundary
The final purchase verdict is deterministically derived:
$$\text{Verdict}(S(p)) = \begin{cases}
\text{Suitable}, & S(p) \ge 0.70 \\
\text{Partly Suitable}, & 0.45 \le S(p) < 0.70 \\
\text{Unsuitable}, & S(p) < 0.45
\end{cases}$$

---

## 3. Virtual Try-On (VTON) Deep-Dive Pipeline

TARZ integrates **IDM-VTON** (*Improving Diffusion Models for Authentic Virtual Try-on in the Wild*), built on Stable Diffusion XL (SDXL).

```
[Raw Garment Image] ──> [MediaPipe Pose Check] ──> [Smart Torso Crop] ──> [768x1024 Canvas] ──┐
                                                                                                │
[User Photo]        ──> [Aspect-Ratio Check]   ──> [Auto-Mask Generator] ──────────────────────┼──> [TryonNet / SDXL Inpaint] ──> [Try-On Result]
                                                                                                │
[Prompt Engine]     ──> [Fabric & Style Tokens] ────────────────────────────────────────────────┘
```

### 3.1 Garment Preprocessing & Extraction Pipeline
A primary failure mode of off-the-shelf VTON models occurs when catalog images contain full-body studio models posing with bags, jewelry, and background props. When fed directly into the model, the cross-attention network transfers background artifacts (such as dried plants, floor textures, or handbags) onto the target user.

TARZ implements an automated pose-guided garment isolation pipeline:
1. **Model Detection:** Evaluates the garment image with `MediaPipe Pose`. If confidence exceeds $0.5$ for shoulder/hip joints, the image is identified as a model photograph.
2. **Torso Bounding:**
   $$y_{\text{top}} = \max\left(0, \min(y_{\text{L\_sh}}, y_{\text{R\_sh}}) - 0.22 \cdot H_{\text{torso}}\right)$$
   $$y_{\text{bottom}} = \min\left(H, \max(y_{\text{L\_hip}}, y_{\text{R\_hip}}) + \kappa \cdot H_{\text{torso}}\right)$$
   where $\kappa = 0.65$ for dresses/kurtis/sarees and $\kappa = 0.25$ for tops/shirts.
3. **Limb Exclusion:** To prevent model arms from bleeding into the garment mask, lateral boundaries are constrained strictly to elbow landmarks:
   $$x_{\text{min}} = \max(0, \min(x_{\text{L\_sh}}, x_{\text{R\_sh}}, x_{\text{L\_el}}, x_{\text{R\_el}}) - 0.05 \cdot W)$$
   $$x_{\text{max}} = \min(W, \max(x_{\text{L\_sh}}, x_{\text{R\_sh}}, x_{\text{L\_el}}, x_{\text{R\_el}}) + 0.05 \cdot W)$$
4. **Resampling:** Letterboxes the cropped garment onto a clean white canvas at $768 \times 1024$ resolution using Lanczos-windowed sinc interpolation (`LANCZOS`).

### 3.2 Dual-Tier Inference & ZeroGPU Quota Failover
Free-tier Hugging Face ZeroGPU allocations impose strict daily compute quotas ($60\text{--}90\text{ s}$ per day). To ensure zero downtime, TARZ incorporates dynamic fallback:

```python
# Automatic Failover Logic in app/tryon.py
try:
    # 1. Primary Priority Execution with Authenticated HF Token
    res = client.predict(..., use_token=True)
except Exception as e:
    if "quota" in str(e).lower() or "exceeded" in str(e).lower():
        # 2. Seamless Anonymous Execution (Token-Free Fallback)
        res = client.predict(..., use_token=False)
```

### 3.3 Root Cause Analysis: The "Three-Arm" Artifact
When users upload photos with crossed or folded arms across their torso, standard inpainting pipelines fail.

```
                    Human Parsing & DensePose Failure on Crossed Arms
                    
     Target Image                    Human Parser Mask              Resulting Artifact
+--------------------+             +--------------------+          +--------------------+
|     \  (o.o)  /    |             |      (   )         |          |     \  (o.o)  /    |
|       / | \        |             |      /   \         |          |       / | \        |
|      [======]      |  =======>   |     ███████        | =====>   |      [======]      | <--- Arm 1 (Crossed)
|       (Arm 1)      |             |     ███████        |          |      | (Arm 2) |   | <--- Arm 2 (Sleeve Inpaint)
|      /   |  \      |             |    (Arm masked     |          |       \  |  /      |
|     /    |   \     |             |     as clothes)    |          |       /  |  \      |
|         (Arm 3)    |             |                    |          |         (Arm 3)    | <--- Arm 3 (Hip Bleedthrough)
+--------------------+             +--------------------+          +--------------------+
```

1. **Semantic Occlusion:** Arms crossed over the chest lie directly within the `upper_clothes` parsing bounding box. The human parser (SCHP / ATR) cannot differentiate between shirt fabric and forearm skin, placing both in the inpainting mask.
2. **DensePose Coordinate Ambiguity:** DensePose fails to map standard UV surface coordinates when arms overlap the thorax.
3. **Diffusion Prior Conflict:** The SDXL diffusion model has a strong prior expectation that sleeves hang vertically from the shoulders. When conditioned on a torso mask, it generates vertical hanging sleeves while simultaneously attempting to retain the crossed arm and wrist keypoints.
4. **Resolution:** Enforced `is_checked_crop = True` to preserve the native 3:4 aspect ratio, preventing horizontal compression that distorts OpenPose joints. Clear UI guidelines instruct users to maintain relaxed arms by their sides.

---

## 4. Neuro-Symbolic Honest Stylist Agent

The **Honest Stylist Agent** combines deterministic recommendation logic with generative natural language generation:

```
[Profile Data] + [Garment Data] + [Scores S(p)] + [Candidate Items]
                            │
                            ▼
           ┌─────────────────────────────────┐
           │   Deterministic Scoring Engine  │
           │      app/scoring.py: score()    │
           └────────────────+────────────────┘
                            │
             Fixed Verdict  │  Structured Evidence JSON
             (Immutable)    ▼
           ┌─────────────────────────────────┐
           │      Honest Stylist Agent       │
           │    app/stylist.py: advise()     │
           │  (Groq GPT-OSS / Gemini Flash)  │
           └────────────────+────────────────┘
                            │
             Raw JSON       ▼
           ┌─────────────────────────────────┐
           │   Regex Extraction & Schema     │
           │          Validator              │
           └────────────────+────────────────┘
                            │
              Valid Output  │  Validation Failure
             ┌──────────────┴──────────────┐
             ▼                             ▼
   [Validated Response]          [Template Engine Fallback]
```

### 4.1 Grounded Evidence Schema
The LLM is prompted with a strict JSON evidence record:
```json
{
  "profile": {
    "gender": "Women",
    "body_shape": "Hourglass",
    "skin": {"depth": "Medium", "undertone": "warm"},
    "recommended_size": "M",
    "fit_confidence": 0.88,
    "approximate": false
  },
  "garment": {
    "id": "p06",
    "name": "Blue Floral Top",
    "category": "Tops",
    "price": 599,
    "colorFamily": "cool",
    "bodyFit": "fitted",
    "styles": ["western", "printed"],
    "occasions": ["casual", "work"]
  },
  "scores": {
    "total": 0.58,
    "terms": {"size": 0.82, "shape": 1.0, "tone": 0.0, "style": 1.0, "occasion": 1.0},
    "weights": {"size": 0.30, "shape": 0.20, "tone": 0.25, "style": 0.15, "occasion": 0.10}
  },
  "verdict": "partly suitable",
  "candidates": [{"id": "p01", "name": "Blue Floral Kurti", "score": 0.84, ...}]
}
```

### 4.2 Guardrail Policies & Hallucination Elimination
* **Verdict Invariance:** The server computes the purchase verdict. The LLM cannot alter this judgment; it can only explain the underlying reasons.
* **Closed-World Candidate Restriction:** The LLM cannot invent product suggestions. Any alternative suggested by the model must exist within `candidates`; unknown IDs are stripped by `_validate()`.
* **Term Attribution:** Every explanation must cite a score term ($s_j$) and its discrete score.
* **Graceful Degradation Cascade:**
  $$\text{Groq (gpt-oss-120b)} \xrightarrow{\text{fail}} \text{Gemini (gemini-2.5-flash)} \xrightarrow{\text{fail}} \text{Deterministic Template}$$

---

## 5. System Architecture & Database Design

### 5.1 Relational Schema (`tarz.db`)
SQLite database schema initialized at startup (`backend/app/db.py`):

```sql
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    created REAL,
    data TEXT,                -- Full JSON profile (measurements, skin, size, shape)
    photo_path TEXT           -- Local filesystem path to uploaded user photo
);

CREATE TABLE IF NOT EXISTS tryons (
    id TEXT PRIMARY KEY,
    created REAL,
    profile_id TEXT,          -- Foreign reference to profiles.id
    product_id TEXT,          -- Reference to catalog product id
    backend TEXT,             -- Inpainting backend engine (e.g. IDM-VTON (HF Space))
    result_url TEXT           -- Serving path (/uploads/tryon_xxx.png or Cloudinary URL)
);

CREATE TABLE IF NOT EXISTS stylist_logs (
    id TEXT PRIMARY KEY,
    created REAL,
    profile_id TEXT,
    product_id TEXT,
    model TEXT,               -- LLM identifier (groq:xxx, gemini:xxx, or template)
    verdict TEXT,             -- Outcome (suitable, partly suitable, unsuitable)
    total REAL                -- Computed score S(p)
);
```

### 5.2 Complete REST API Specification

| HTTP Verb | Path | Request Payload | Response Schema | Description |
|---|---|---|---|---|
| `GET` | `/api/config` | *None* | `{"groq": bool, "gemini": bool, "huggingface": bool, ...}` | System readiness and active API credentials |
| `GET` | `/api/catalog` | Query params: `gender`, `category`, `occasion` | `[Product]` | Catalog search and filtering |
| `POST` | `/api/analyze` | `multipart/form-data`: `photo` (file), `gender`, `height_cm`, optional manual measurements | `{"profile_id": str, "body_shape": str, "skin": {...}, "size": {...}}` | Computer vision pose analysis and sizing |
| `POST` | `/api/recommend` | `application/json`: `profile_id`, `preferences`, `weights`, `limit` | `{"items": [{"id": ..., "score": {"total": float, "terms": {...}}}]}` | Multi-criteria catalog ranking |
| `POST` | `/api/tryon` | `application/x-www-form-urlencoded`: `profile_id`, `product_id` | `{"url": str, "backend": str, "errors": [str]}` | Garment prep and virtual try-on execution |
| `POST` | `/api/stylist` | `application/json`: `profile_id`, `product_id`, `question`, `preferences` | `{"verdict": str, "headline": str, "reasons": [str], "alternatives": [...]}` | Guardrailed conversational styling analysis |
| `DELETE` | `/api/profile/{id}` | Path param: `id` | `{"deleted": true}` | Privacy compliance: purges profile and user photo |

---

## 6. Directory Structure

```
tarz-ai-fashion/
│
├── README.md                           # Master Academic & Technical Specification
├── .gitignore                          # Git exclusion (env, db, uploads, node_modules)
│
├── backend/
│   ├── .env.example                    # Environment variable template
│   ├── .env                            # Local private API keys (never commit)
│   ├── requirements.txt                # Python package manifest
│   │
│   ├── app/
│   │   ├── __init__.py                 # Module initializer
│   │   ├── main.py                     # FastAPI application endpoints & routing
│   │   ├── config.py                   # Environment loader & timeout constants
│   │   ├── db.py                       # SQLite database manager & connection pool
│   │   ├── body_agent.py               # MediaPipe Pose & skin colorimetry engine
│   │   ├── catalog.py                  # In-memory garment catalog store
│   │   ├── scoring.py                  # Multi-criteria recommendation engine
│   │   ├── tryon.py                    # VTON orchestration, auto-crop & failover
│   │   ├── stylist.py                  # Guardrailed LLM stylist (Groq / Gemini)
│   │   └── storage.py                  # Image upload handler & Cloudinary mirror
│   │
│   ├── data/
│   │   ├── catalog.json                # Structured catalog (Kurtis, Tops, Sarees, Dresses)
│   │   └── tarz.db                     # SQLite database file
│   │
│   ├── scripts/
│   │   └── refresh_images.py           # Optional catalog image updater via Pexels
│   │
│   ├── tests/
│   │   ├── conftest.py                 # Pytest configuration
│   │   └── test_core.py                # Unit and integration test suite
│   │
│   └── uploads/                        # Generated try-on images & temporary files
│
└── frontend/
    ├── package.json                    # Node.js dependencies
    ├── vite.config.js                  # Vite configuration
    ├── index.html                      # Single-page application root
    ├── .env.example                    # Frontend environment template
    ├── .env                            # VITE_API_URL specification
    │
    └── src/
        ├── main.jsx                    # React entrypoint
        ├── App.jsx                     # Top-level screen navigation controller
        ├── api.js                      # Axios/Fetch API client wrapper
        ├── styles.css                  # UI design system styles
        └── screens/
            ├── Landing.jsx             # Hero view & workflow introduction
            ├── Analyze.jsx             # Photo upload & manual measurement form
            ├── Catalog.jsx             # Smart catalog with real-time scoring
            ├── TryOn.jsx               # Virtual try-on side-by-side view
            └── Stylist.jsx             # Explainable stylist consultation interface
```

---

## 7. Installation, Configuration & Local Deployment

### 7.1 Prerequisites
* **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+ recommended)
* **Python Runtime:** Python 3.10, 3.11, or 3.12 (64-bit)
* **Node.js Runtime:** Node.js 18.x or 20.x with `npm`

### 7.2 Backend Setup
Open a terminal in the project directory:

```bash
# 1. Navigate to backend directory
cd backend

# 2. (Optional but recommended) Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Configure environment variables
# Copy template and edit keys:
cp .env.example .env
```

#### Environment Variables (`backend/.env`)
```ini
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
HF_TOKEN=hf_your_huggingface_token_here
REPLICATE_API_TOKEN=
PEXELS_API_KEY=
CLOUDINARY_URL=
FRONTEND_ORIGIN=http://localhost:5173
```

#### Launch Backend Server
```bash
uvicorn app.main:app --reload --port 8000
```
*API documentation and Swagger UI will be available at `http://127.0.0.1:8000/docs`.*

### 7.3 Frontend Setup
Open a second terminal window:

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Configure frontend environment
echo VITE_API_URL=http://localhost:8000 > .env

# 3. Install NPM packages
npm install

# 4. Start local development server
npm run dev
```
*Access the user interface at `http://localhost:5173`.*

### 7.4 Running Test Suite
Execute the integration test suite (runs 100% locally with zero external network overhead):
```bash
cd backend
python -m pytest -q tests
```

---

## 8. Experimental Methodology & Evaluation Guide

This section provides recommendations for researchers evaluating this framework:

### 8.1 Metric Framework
1. **Perceptual Quality (Virtual Try-On):**
   * **Fréchet Inception Distance (FID):** Evaluates generative image distribution quality compared to real garment-model pairings.
   * **Learned Perceptual Image Patch Similarity (LPIPS):** Quantifies perceptual garment detail preservation (prints, ruffles, lace).
   * **Structural Similarity Index (SSIM):** Compares unmasked skin and background preservation against the original photo.
2. **Recommendation Accuracy:**
   * **Mean Reciprocal Rank (MRR) & NDCG@K:** Evaluates whether top-ranked items ($S(p)$) align with expert stylist ground truth.
   * **Fit Accuracy:** Percentage agreement between computed size $S^*$ and verified customer fit data.
3. **Conversational Explainability:**
   * **Faithfulness / Grounding Score:** Evaluates whether reasons produced by the LLM match the underlying mathematical terms ($s_j$).
   * **Hallucination Rate:** Frequency of non-existent product attributes, fabricated alternative IDs, or unauthorized verdict modifications (baseline targets: $0.0\%$).

### 8.2 Failure Modes & Edge Cases
* **Complex Self-Occluding Poses:** As analyzed in Section 3.3, crossed arms across the torso cause multi-arm artifacts across diffusion inpainting models. Future iterations should incorporate 3D human mesh estimation (e.g., SMPL-X) to disentangle foreground limbs before 2D inpainting.
* **Uncalibrated Monocular Depth:** Calculating 3D body measurements from a single 2D image is an ill-posed inverse problem. Scale estimation relies on known user height ($H$); optical perspective distortion can affect bust/hip estimations if the camera is tilted.
* **Complex Multi-Layered Garments:** Full-length Indian ethnic sarees and layered dupattas present challenging draping dynamics that standard upper-body try-on pipelines approximate as dresses. Dedicated multi-garment segmentation networks are required for full drape synthesis.

---

## 9. Ethics, Privacy & Responsible AI

* **Data Minimization:** No personal data or biometric photographs are permanently retained without user intent.
* **User Agency & Deletion Rights:** The `DELETE /api/profile/{id}` endpoint deletes both the SQLite profile metadata and the local/cloud photograph immediately upon user request ("Delete my data").
* **Honest Guardrails (No Deceptive Flattery):** Commercial recommendation algorithms frequently prioritize conversion over accurate fit. The Honest Stylist Agent is explicitly constrained to communicate when a garment is unsuitable or ill-fitting based on verified mathematical thresholds.

---

## 10. License & Citation

This project is released under the **MIT License**.

If you use TARZ in your research or academic publications, please cite:

```bibtex
@article{tarz2026fashion,
  title={TARZ: A Multi-Modal Neuro-Symbolic Decision Support and Virtual Try-On Framework for Fashion E-Commerce},
  author={TARZ Research and Development Team},
  year={2026},
  journal={arXiv preprint},
  url={https://github.com/your-username/tarz-ai-fashion}
}
```