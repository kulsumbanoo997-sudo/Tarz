import io
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app import body_agent as ba, catalog, scoring, stylist
from app.main import app

client = TestClient(app)


def test_body_shape_bands():
    assert ba.classify_body_shape(1.25) == "Inverted Triangle"
    assert ba.classify_body_shape(1.10) == "Apple"
    assert ba.classify_body_shape(1.00) == "Hourglass"
    assert ba.classify_body_shape(0.90) == "Rectangle"
    assert ba.classify_body_shape(0.70) == "Pear"


def test_skin_rules():
    assert ba.classify_skin((200, 170, 150))["undertone"] == "warm"
    assert ba.classify_skin((180, 170, 175))["undertone"] == "cool"
    assert ba.classify_skin((100, 80, 70))["depth"] == "Deep"
    assert ba.classify_skin((230, 200, 190))["depth"] == "Fair"


def test_size_and_confidence():
    r = ba.recommend_size({"bust": 91, "waist": 73, "hip": 97}, "Women")
    assert r["recommended"] == "M" and r["fit_confidence"] == 1.0
    r2 = ba.recommend_size({"bust": 91, "hip": 110}, "Women")  # hip far from bust size
    assert 0 < r2["fit_confidence"] < 1.0
    assert ba.fit_confidence({}) == 0.0


def test_weights_normalise_and_score_range():
    w = scoring.normalise({"size": 3, "shape": 1})
    assert abs(sum(w.values()) - 1) < 1e-9
    prof = ba.finish_profile(ba.analyze_manual("Women", 91, 73, 97, undertone="warm"))
    ranked = scoring.rank(prof, catalog.load())
    assert ranked and all(0 <= r["score"]["total"] <= 1 for r in ranked)
    assert ranked == sorted(ranked, key=lambda r: (-r["score"]["total"], r["price"]))
    assert all(r["gender"] == "Women" for r in ranked)


def test_prototype_rule_is_special_case():
    """With only shape+tone weights (1/3, 2/3) the score equals the prototype rule r(p)/3."""
    prof = ba.finish_profile(ba.analyze_manual("Women", 91, 73, 97, undertone="warm"))
    prof["body_shape"] = "Apple"
    for p in catalog.load():
        if p["gender"] != "Women":
            continue
        s = scoring.score_product(prof, p, {}, {"size": 0, "shape": 1, "tone": 2, "style": 0, "occasion": 0})["total"]
        r = 2 * (p["colorFamily"] == "warm") + (p["bodyFit"] in ("aline", "flowy"))
        assert abs(s - r / 3) < 1e-3


def test_stylist_validation_blocks_invented_ids():
    prof = ba.finish_profile(ba.analyze_manual("Women", 91, 73, 97, undertone="cool"))
    prod = catalog.load()[0]
    sc = scoring.score_product(prof, prod)
    cands = [c for c in scoring.rank(prof, catalog.load()) if c["id"] != prod["id"]][:6]
    ev = stylist.build_evidence(prof, prod, sc, cands, {}, "")
    raw = {"headline": "x", "reasons": ["Tone (1): ok"], "alternatives": [{"id": "ZZZ", "why": "fake"}, {"id": cands[0]["id"], "why": "real"}]}
    out = stylist._validate(raw, ev)
    assert [a["id"] for a in out["alternatives"]] == [cands[0]["id"]]
    assert stylist._extract_json("```json\n{\"a\":1}\n```") == {"a": 1}


def test_api_flow_manual():
    cat = client.get("/api/catalog", params={"gender": "Men"}).json()
    assert len(cat) == 8
    r = client.post("/api/analyze", data={"gender": "Women", "bust_cm": 91, "waist_cm": 73, "hip_cm": 97, "undertone": "warm"})
    assert r.status_code == 200, r.text
    pid = r.json()["profile_id"]
    rec = client.post("/api/recommend", json={"profile_id": pid, "preferences": {"styles": ["ethnic"], "occasion": "festive"}, "limit": 5}).json()
    assert len(rec["items"]) == 5
    pr = rec["items"][-1]["id"]
    st = client.post("/api/stylist", json={"profile_id": pid, "product_id": pr, "question": "Can I wear this to a wedding?"}).json()
    assert (st["model"] == "template" or st["model"].startswith("groq:")) and st["verdict"] in ("suitable", "partly suitable", "unsuitable")
    assert client.post("/api/tryon", data={"profile_id": pid, "product_id": pr}).status_code == 400  # no photo
    assert client.delete(f"/api/profile/{pid}").json()["deleted"]
    assert client.post("/api/recommend", json={"profile_id": pid}).status_code == 404


def test_api_validation():
    assert client.post("/api/analyze", data={"gender": "X"}).status_code == 400
    assert client.post("/api/analyze", data={"gender": "Women"}).status_code == 422  # no photo, no measurements
    assert client.post("/api/analyze", data={"gender": "Women", "bust_cm": 10, "hip_cm": 90}).status_code == 422


def test_api_does_not_enable_cors():
    r = client.get("/api/health", headers={"Origin": "https://tarz-seven.vercel.app"})
    assert "access-control-allow-origin" not in r.headers


def test_photo_without_person_is_rejected():
    buf = io.BytesIO()
    Image.fromarray(np.full((480, 360, 3), 235, np.uint8)).save(buf, "JPEG")
    r = client.post("/api/analyze", data={"gender": "Women", "height_cm": 160}, files={"photo": ("a.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 422 and "No person" in r.json()["detail"]
