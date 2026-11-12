"""Honest Stylist Agent: LLM explanation grounded in server-computed evidence.

The server decides the verdict from the weighted score; the LLM only explains it and may
pick alternatives ONLY from a candidate list. Output is validated; on any failure a
template explanation built from the same evidence is returned.
"""
import json
import re
from typing import Optional

import httpx

from . import config
from .scoring import verdict_for

SYSTEM = """You are TARZ's Honest Stylist, advising a shopper about ONE garment.
Rules:
- Use ONLY the facts in the JSON you receive. Never invent measurements, fabrics, prices or products.
- Be honest: if the garment is a poor match, say so plainly and kindly. Do not flatter.
- The "verdict" was already decided by the server. Explain it; do not change it.
- Every reason must refer to a named score term (size, shape, tone, style, occasion) and its value (1 = match, 0 = mismatch, 0.5 = unknown).
- Suggest alternatives ONLY by choosing ids from "candidates" (at most 3, each with a short reason).
- Sizes are estimates. If profile.approximate is true, mention that measurements come from a photo and may be off.
- If "question" is not empty, also answer it briefly in "answer", using only the given facts.
Return ONLY a JSON object, no markdown:
{"headline": str, "reasons": [str], "tips": [str], "alternatives": [{"id": str, "why": str}], "answer": str}"""

TERM_TEXT = {
    "size": ("fits your estimated size well", "may not fit your measurements closely"),
    "shape": ("suits your body shape", "is not the usual best match for your body shape"),
    "tone": ("is in your suggested colour family", "is outside your suggested colour family"),
    "style": ("matches the styles you chose", "is different from the styles you chose"),
    "occasion": ("works for your chosen occasion", "is not tagged for your chosen occasion"),
}


def build_evidence(profile: dict, product: dict, score: dict, candidates: list, prefs: dict, question: str) -> dict:
    return {
        "profile": {
            "gender": profile["gender"], "body_shape": profile["body_shape"],
            "skin": {k: (profile.get("skin") or {}).get(k) for k in ("depth", "undertone")},
            "recommended_size": (profile.get("size") or {}).get("recommended"),
            "fit_confidence": (profile.get("size") or {}).get("fit_confidence"),
            "approximate": bool(profile.get("approximate")),
        },
        "garment": {k: product[k] for k in ("id", "name", "category", "price", "colorFamily", "bodyFit", "styles", "occasions")},
        "preferences": prefs or {},
        "scores": {"total": score["total"], "terms": score["terms"], "weights": score["weights"]},
        "verdict": verdict_for(score["total"]),
        "candidates": [
            {"id": c["id"], "name": c["name"], "category": c["category"], "price": c["price"], "score": c["score"]["total"],
             "colorFamily": c["colorFamily"], "bodyFit": c["bodyFit"]} for c in candidates
        ],
        "question": (question or "").strip()[:300],
    }


def template_answer(ev: dict) -> dict:
    terms = ev["scores"]["terms"]
    reasons = []
    for k, v in terms.items():
        if k in ("style", "occasion") and v == 0.5:
            continue
        pos, neg = TERM_TEXT[k]
        if v >= 0.75:
            reasons.append(f"{k.capitalize()} ({v:g}): this garment {pos}.")
        elif v <= 0.25:
            reasons.append(f"{k.capitalize()} ({v:g}): this garment {neg}.")
        elif k == "size":
            reasons.append(f"Size ({v:g}): fit is only moderately close to your estimated size.")
    head = {"suitable": "Good match for you.", "partly suitable": "A partial match; read the notes below.",
            "unsuitable": "Honestly, this is probably not the best pick for you."}[ev["verdict"]]
    tips = []
    if ev["profile"]["approximate"]:
        tips.append("Your measurements were estimated from a photo, so check the size chart before buying.")
    alts = [{"id": c["id"], "why": f"Higher overall score ({c['score']:.2f}) for your profile."} for c in ev["candidates"][:3]
            if ev["verdict"] != "suitable" or c["score"] > ev["scores"]["total"]]
    return {"headline": head, "reasons": reasons or ["Not enough information to explain this score."],
            "tips": tips, "alternatives": alts[:3], "answer": ""}


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except Exception:
        return None


def _validate(raw: Optional[dict], ev: dict) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None
    ids = {c["id"] for c in ev["candidates"]}
    reasons = [str(r)[:300] for r in (raw.get("reasons") or []) if isinstance(r, str)][:5]
    tips = [str(t)[:300] for t in (raw.get("tips") or []) if isinstance(t, str)][:3]
    alts = []
    for a in raw.get("alternatives") or []:
        if isinstance(a, dict) and a.get("id") in ids and len(alts) < 3:
            alts.append({"id": a["id"], "why": str(a.get("why", ""))[:200]})
    if not reasons or not isinstance(raw.get("headline"), str):
        return None
    return {"headline": raw["headline"][:200], "reasons": reasons, "tips": tips, "alternatives": alts,
            "answer": str(raw.get("answer") or "")[:600]}


async def _groq(ev: dict) -> Optional[dict]:
    if not config.GROQ_API_KEY:
        return None
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(ev, ensure_ascii=False)}]
    async with httpx.AsyncClient(timeout=40) as cx:
        for model in dict.fromkeys(config.GROQ_MODELS):
            try:
                r = await cx.post("https://api.groq.com/openai/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
                                  json={"model": model, "messages": msgs, "temperature": 0.3, "max_completion_tokens": 1500})
                if r.status_code != 200:
                    continue
                out = _validate(_extract_json(r.json()["choices"][0]["message"].get("content", "")), ev)
                if out:
                    return {**out, "model": f"groq:{model}"}
            except Exception:
                continue
    return None


async def _gemini(ev: dict) -> Optional[dict]:
    if not config.GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(ev, ensure_ascii=False)}]}],
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}}
    try:
        async with httpx.AsyncClient(timeout=40) as cx:
            r = await cx.post(url, params={"key": config.GEMINI_API_KEY}, json=body)
        if r.status_code != 200:
            return None
        txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        out = _validate(_extract_json(txt), ev)
        return {**out, "model": f"gemini:{config.GEMINI_MODEL}"} if out else None
    except Exception:
        return None


async def advise(profile, product, score, candidates, prefs=None, question="") -> dict:
    ev = build_evidence(profile, product, score, candidates, prefs or {}, question)
    out = await _groq(ev) or await _gemini(ev)
    if out is None:
        out = {**template_answer(ev), "model": "template"}
    # the server's verdict always wins
    return {"verdict": ev["verdict"], "score": ev["scores"], **out}
