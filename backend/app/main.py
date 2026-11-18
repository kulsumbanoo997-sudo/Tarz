import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import body_agent, catalog, config, db, scoring, storage, stylist, tryon

app = FastAPI(title="TARZ API", version="1.0")
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")
db.init()

VALID_SHAPES = list(scoring.BODY_FITS)


def _profile_or_404(pid: str):
    prof, photo = db.get_profile(pid)
    if not prof:
        raise HTTPException(404, "Profile not found. Please analyse again.")
    return prof, photo


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def cfg():
    """Which services are configured (never returns keys)."""
    return {"groq": bool(config.GROQ_API_KEY), "gemini": bool(config.GEMINI_API_KEY), "huggingface": bool(config.HF_TOKEN),
            "replicate": bool(config.REPLICATE_API_TOKEN), "cloudinary": bool(config.CLOUDINARY_URL),
            "weights": scoring.DEFAULT_WEIGHTS, "occasions": ["casual", "work", "party", "festive", "wedding", "formal", "vacation"],
            "styles": ["ethnic", "western", "formal", "printed", "solid"]}


@app.get("/api/catalog")
def get_catalog(gender: Optional[str] = None, category: Optional[str] = None, occasion: Optional[str] = None,
                min_price: Optional[int] = None, max_price: Optional[int] = None):
    f = {"gender": gender, "category": category, "occasion": occasion, "strict_occasion": bool(occasion),
         "min_price": min_price, "max_price": max_price}
    return [p for p in catalog.load() if scoring.passes_filters(p, f)]


@app.post("/api/analyze")
async def analyze(gender: str = Form(...), height_cm: Optional[float] = Form(None), photo: Optional[UploadFile] = File(None),
                  bust_cm: Optional[float] = Form(None), waist_cm: Optional[float] = Form(None), hip_cm: Optional[float] = Form(None),
                  body_shape: Optional[str] = Form(None), undertone: Optional[str] = Form(None), depth: Optional[str] = Form(None)):
    if gender not in ("Women", "Men"):
        raise HTTPException(400, "gender must be Women or Men")
    if body_shape and body_shape not in VALID_SHAPES:
        raise HTTPException(400, "invalid body_shape")
    if undertone and undertone not in ("warm", "cool", "neutral"):
        raise HTTPException(400, "invalid undertone")
    photo_path = None
    try:
        if photo is not None and photo.filename:
            data = await photo.read()
            if len(data) > 12 * 1024 * 1024:
                raise HTTPException(413, "Photo too large (max 12 MB).")
            prof = body_agent.analyze_photo(data, gender, height_cm)
            # manual overrides win over automatic estimates
            if body_shape:
                prof["body_shape"] = body_shape
            if undertone:
                prof["skin"] = {**(prof.get("skin") or {}), "undertone": undertone, "depth": depth or (prof.get("skin") or {}).get("depth", "Medium")}
            for k, v in (("bust", bust_cm), ("waist", waist_cm), ("hip", hip_cm)):
                if v:
                    prof["measurements"][k] = v
                    prof["approximate"] = False
            img = body_agent.load_image(data)
            import io
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "JPEG", quality=90)
            photo_path, _ = storage.save_local(buf.getvalue(), "photo")
        else:
            prof = body_agent.analyze_manual(gender, bust_cm, waist_cm, hip_cm, height_cm, body_shape, undertone or "neutral", depth or "Medium")
    except body_agent.AnalysisError as e:
        raise HTTPException(422, str(e))
    prof = body_agent.finish_profile(prof)
    pid = db.save_profile(prof, photo_path)
    return {"profile_id": pid, "profile": prof, "has_photo": bool(photo_path)}


class Prefs(BaseModel):
    styles: list[str] = []
    occasion: Optional[str] = None


class RecommendReq(BaseModel):
    profile_id: str
    preferences: Prefs = Prefs()
    category: Optional[str] = None
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    strict_occasion: bool = False
    weights: Optional[dict] = None
    limit: Optional[int] = None


@app.post("/api/recommend")
def recommend(req: RecommendReq):
    prof, _ = _profile_or_404(req.profile_id)
    filters = {"gender": prof["gender"], "category": req.category, "occasion": req.preferences.occasion,
               "strict_occasion": req.strict_occasion, "min_price": req.min_price, "max_price": req.max_price}
    return {"items": scoring.rank(prof, catalog.load(), req.preferences.model_dump(), filters, req.weights, req.limit)}


class StylistReq(BaseModel):
    profile_id: str
    product_id: str
    preferences: Prefs = Prefs()
    question: str = ""


@app.post("/api/stylist")
async def stylist_endpoint(req: StylistReq):
    prof, _ = _profile_or_404(req.profile_id)
    prod = catalog.get(req.product_id)
    if not prod:
        raise HTTPException(404, "Product not found")
    prefs = req.preferences.model_dump()
    sc = scoring.score_product(prof, prod, prefs)
    ranked = scoring.rank(prof, catalog.load(), prefs, {"gender": prof["gender"]}, None, None)
    cands = [c for c in ranked if c["id"] != prod["id"]][:6]
    out = await stylist.advise(prof, prod, sc, cands, prefs, req.question)
    out["alternatives"] = [{**a, "product": next((c for c in cands if c["id"] == a["id"]), None)} for a in out["alternatives"]]
    db.log_stylist(req.profile_id, prod["id"], out["model"], out["verdict"], sc["total"])
    return out


@app.post("/api/tryon")
async def tryon_endpoint(profile_id: str = Form(...), product_id: str = Form(...)):
    prof, photo = _profile_or_404(profile_id)
    prod = catalog.get(product_id)
    if not prod:
        raise HTTPException(404, "Product not found")
    if not photo or not os.path.exists(photo):
        raise HTTPException(400, "Try-on needs a photo. Analyse with a photo first.")
    try:
        res = await tryon.run_tryon(photo, prod)
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    mirror = None
    if config.CLOUDINARY_URL:
        with open(res["path"], "rb") as f:
            mirror = storage.cloudinary_upload(f.read(), os.path.basename(res["path"]).split(".")[0])
    db.log_tryon(profile_id, product_id, res["backend"], mirror or res["url"])
    return {"url": mirror or res["url"], "backend": res["backend"], "errors": res["errors"]}


@app.delete("/api/profile/{pid}")
def delete_profile(pid: str):
    path = db.delete_profile(pid)
    if path and os.path.exists(path):
        os.remove(path)
    return {"deleted": True}
