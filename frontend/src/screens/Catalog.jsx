import React, { useEffect, useState } from "react";
import { recommend, img, deleteProfile } from "../api.js";

const TERMS = [["size", "Size"], ["shape", "Shape"], ["tone", "Tone"], ["style", "Style"], ["occasion", "Occasion"]];
const VCLASS = { suitable: "good", "partly suitable": "mid", unsuitable: "bad" };

function ProfileCard({ profile, hasPhoto, onDelete }) {
  const m = profile.measurements || {}, s = profile.size || {}, k = profile.skin;
  return (
    <aside className="card profile">
      <h3>Your profile</h3>
      <dl>
        <dt>Body shape</dt><dd>{profile.body_shape}</dd>
        <dt>Skin</dt><dd>{k ? `${k.depth}, ${k.undertone} undertone` : "—"}</dd>
        <dt>Size</dt><dd>{s.recommended || "—"} {s.recommended && <small>(fit confidence {Math.round((s.fit_confidence || 0) * 100)}%)</small>}</dd>
        {m.bust && <><dt>Bust / chest</dt><dd>{m.bust} cm</dd></>}
        {m.waist && <><dt>Waist</dt><dd>{m.waist} cm</dd></>}
        {m.hip && <><dt>Hip</dt><dd>{m.hip} cm</dd></>}
      </dl>
      {profile.approximate && <p className="hint">Estimated from a photo, so treat as approximate.</p>}
      {(profile.warnings || []).map((w, i) => <p key={i} className="hint">{w}</p>)}
      <button className="link" onClick={onDelete}>Delete my data</button>
      {!hasPhoto && <p className="hint">No photo saved: try-on needs a photo.</p>}
    </aside>
  );
}

export default function Catalog({ go, profileId, profile, hasPhoto, prefs, cfg, reset }) {
  const [items, setItems] = useState([]);
  const [category, setCategory] = useState("");
  const [maxPrice, setMaxPrice] = useState("");
  const [strict, setStrict] = useState(false);
  const [open, setOpen] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    recommend({ profile_id: profileId, preferences: prefs, category: category || null, max_price: maxPrice ? +maxPrice : null, strict_occasion: strict })
      .then((r) => { setItems(r.items); setErr(""); })
      .catch((e) => setErr(e.message)).finally(() => setLoading(false));
  }, [profileId, prefs, category, maxPrice, strict]);

  const cats = [...new Set(items.map((i) => i.category))];
  const del = async () => { if (confirm("Delete your photo and profile from the server?")) { try { await deleteProfile(profileId); } catch {} reset(); } };
  const w = cfg?.weights;

  return (
    <div className="layout">
      <ProfileCard profile={profile} hasPhoto={hasPhoto} onDelete={del} />
      <section>
        <div className="filters">
          <select value={category} onChange={(e) => setCategory(e.target.value)}><option value="">All categories</option>{cats.map((c) => <option key={c}>{c}</option>)}</select>
          <input type="number" placeholder="Max price ₹" value={maxPrice} onChange={(e) => setMaxPrice(e.target.value)} />
          {prefs.occasion && <label className="inline"><input type="checkbox" checked={strict} onChange={(e) => setStrict(e.target.checked)} /> only “{prefs.occasion}”</label>}
          <button onClick={() => go("analyze")}>Edit profile</button>
        </div>
        {err && <div className="banner err">{err}</div>}
        {loading && <p className="hint">Ranking…</p>}
        <div className="grid">
          {items.map((p) => (
            <article key={p.id} className="item">
              <div className="imgwrap"><img src={img(p.image)} alt={p.name} loading="lazy" />
                <span className={"badge " + VCLASS[p.verdict]}>{Math.round(p.score.total * 100)}%</span></div>
              <h4>{p.name}</h4>
              <div className="meta">₹{p.price} <s>₹{p.mrp}</s> · {p.category}</div>
              <button className="link" onClick={() => setOpen(open === p.id ? null : p.id)}>{open === p.id ? "Hide" : "Why this score?"}</button>
              {open === p.id && (
                <div className="terms">
                  {TERMS.map(([k, l]) => (
                    <div key={k} className="term"><span>{l} <small>w={w ? w[k] : ""}</small></span>
                      <div className="bar"><i style={{ width: p.score.terms[k] * 100 + "%" }} /></div><b>{p.score.terms[k]}</b></div>))}
                  <small>1 = match, 0 = mismatch, 0.5 = unknown / no preference</small>
                </div>)}
              <div className="row">
                <button onClick={() => go("tryon", p)}>Try on</button>
                <button className="primary" onClick={() => go("stylist", p)}>Ask stylist</button>
              </div>
            </article>))}
        </div>
      </section>
    </div>
  );
}
