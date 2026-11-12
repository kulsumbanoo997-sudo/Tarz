"""Optional: replace catalog images with Pexels search results (needs PEXELS_API_KEY in .env).

Usage (from backend/):  python scripts/refresh_images.py
Writes data/image_overrides.json. Delete that file to go back to the original images.
Review the results: search matches are approximate, so garment names may not match photos exactly.
"""
import json, sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

if not config.PEXELS_API_KEY:
    sys.exit("PEXELS_API_KEY is empty. Add it to backend/.env first.")

items = json.load(open(config.DATA_DIR / "catalog.json", encoding="utf8"))
used, out = set(), {}
with httpx.Client(timeout=20, headers={"Authorization": config.PEXELS_API_KEY}) as cx:
    for p in items:
        q = f"{p['name']} {'women' if p['gender']=='Women' else 'men'} fashion"
        r = cx.get("https://api.pexels.com/v1/search", params={"query": q, "per_page": 8, "orientation": "portrait"})
        if r.status_code != 200:
            print(p["id"], "HTTP", r.status_code); continue
        for ph in r.json().get("photos", []):
            url = ph["src"]["large"]
            if url not in used:
                used.add(url); out[p["id"]] = url; print(p["id"], p["name"], "->", ph["photographer"]); break
json.dump(out, open(config.DATA_DIR / "image_overrides.json", "w"), indent=1)
print(f"wrote {len(out)} overrides. Pexels requires crediting photographers/Pexels where photos are shown.")
