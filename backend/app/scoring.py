"""Smart Catalog: weighted, explainable scoring  S(p) = sum_j w_j * s_j(p)."""
from .body_agent import size_distances, fit_confidence

DEFAULT_WEIGHTS = {"size": 0.30, "shape": 0.20, "tone": 0.25, "style": 0.15, "occasion": 0.10}
BODY_FITS = {
    "Hourglass": ["fitted", "structured"],
    "Pear": ["structured", "fitted"],
    "Apple": ["aline", "flowy"],
    "Rectangle": ["structured", "fitted"],
    "Inverted Triangle": ["aline", "flowy"],
}
# How strictly a silhouette penalises a size mismatch (flowy garments forgive more).
FIT_STRICTNESS = {"fitted": 1.5, "structured": 1.5, "aline": 1.0, "flowy": 0.6}
VERDICT_THRESHOLDS = {"suitable": 0.70, "partly": 0.45}  # tunable, not validated


def normalise(weights):
    w = {k: max(0.0, float((weights or {}).get(k, DEFAULT_WEIGHTS[k]))) for k in DEFAULT_WEIGHTS}
    t = sum(w.values()) or 1.0
    return {k: v / t for k, v in w.items()}


def term_values(profile: dict, product: dict, prefs: dict) -> dict:
    meas = {k: v for k, v in profile["measurements"].items() if k in ("bust", "waist", "hip")}
    rec = (profile.get("size") or {}).get("recommended")
    if meas and rec:
        c = fit_confidence(size_distances(meas, profile["gender"], rec))
        s_size = max(0.0, 1 - min(1.0, FIT_STRICTNESS.get(product["bodyFit"], 1.0) * (1 - c)))
    else:
        s_size = 0.5  # unknown -> neutral
    s_shape = 1.0 if product["bodyFit"] in BODY_FITS.get(profile["body_shape"], []) else 0.0
    under = (profile.get("skin") or {}).get("undertone")
    s_tone = 1.0 if under and product["colorFamily"] == under else (0.5 if not under else 0.0)
    styles = prefs.get("styles") or []
    s_style = 0.5 if not styles else (1.0 if set(styles) & set(product.get("styles", [])) else 0.0)
    occ = prefs.get("occasion")
    s_occ = 0.5 if not occ else (1.0 if occ in product.get("occasions", []) else 0.0)
    return {"size": round(s_size, 3), "shape": s_shape, "tone": s_tone, "style": s_style, "occasion": s_occ}


def score_product(profile, product, prefs=None, weights=None) -> dict:
    prefs, w = prefs or {}, normalise(weights)
    terms = term_values(profile, product, prefs)
    total = sum(w[k] * terms[k] for k in w)
    return {"total": round(total, 3), "terms": terms, "weights": {k: round(v, 3) for k, v in w.items()}}


def verdict_for(total: float) -> str:
    if total >= VERDICT_THRESHOLDS["suitable"]:
        return "suitable"
    if total >= VERDICT_THRESHOLDS["partly"]:
        return "partly suitable"
    return "unsuitable"


def passes_filters(p: dict, f: dict) -> bool:
    f = f or {}
    if f.get("gender") and p["gender"] != f["gender"]:
        return False
    if f.get("category") and p["category"] != f["category"]:
        return False
    if f.get("occasion") and f.get("strict_occasion") and f["occasion"] not in p.get("occasions", []):
        return False
    if f.get("min_price") is not None and p["price"] < f["min_price"]:
        return False
    if f.get("max_price") is not None and p["price"] > f["max_price"]:
        return False
    return True


def rank(profile, catalog, prefs=None, filters=None, weights=None, limit=None):
    out = []
    for p in catalog:
        if not passes_filters(p, filters or {"gender": profile["gender"]}):
            continue
        s = score_product(profile, p, prefs, weights)
        out.append({**p, "score": s, "verdict": verdict_for(s["total"])})
    out.sort(key=lambda x: (-x["score"]["total"], x["price"]))
    return out[:limit] if limit else out
