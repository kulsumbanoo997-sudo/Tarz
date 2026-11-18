import React, { useState } from "react";
import { analyze } from "../api.js";

export default function Analyze({ cfg, prefs, onProfile, apiReady, apiConnecting }) {
  const [mode, setMode] = useState("photo");
  const [gender, setGender] = useState("Women");
  const [f, setF] = useState({ height: "", bust: "", waist: "", hip: "", shape: "", undertone: "", depth: "" });
  const [file, setFile] = useState(null);
  const [p, setP] = useState(prefs);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  const toggle = (s) => setP({ ...p, styles: p.styles.includes(s) ? p.styles.filter((x) => x !== s) : [...p.styles, s] });

  const submit = async () => {
    setErr("");
    if (mode === "photo" && !file) return setErr("Please choose a photo.");
    if (mode === "photo" && !f.height) return setErr("Enter your height so measurements can be estimated.");
    const fd = new FormData();
    fd.append("gender", gender);
    if (f.height) fd.append("height_cm", f.height);
    if (mode === "photo") fd.append("photo", file);
    ["bust", "waist", "hip"].forEach((k) => f[k] && fd.append(k + "_cm", f[k]));
    if (f.shape) fd.append("body_shape", f.shape);
    if (f.undertone) fd.append("undertone", f.undertone);
    if (f.depth) fd.append("depth", f.depth);
    setBusy(true);
    try { onProfile(await analyze(fd), p); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  const shapes = ["Hourglass", "Pear", "Apple", "Rectangle", "Inverted Triangle"];

  return (
    <section className="card narrow">
      <h2>Upload &amp; Analyze</h2>
      <div className="tabs">
        <button className={mode === "photo" ? "on" : ""} onClick={() => setMode("photo")}>Photo</button>
        <button className={mode === "manual" ? "on" : ""} onClick={() => setMode("manual")}>Manual measurements</button>
      </div>
      <div className="grid2">
        <label>Gender<select value={gender} onChange={(e) => setGender(e.target.value)}><option>Women</option><option>Men</option></select></label>
        <label>Height (cm){mode === "photo" ? " *" : ""}<input type="number" value={f.height} onChange={set("height")} placeholder="e.g. 162" /></label>
      </div>
      {mode === "photo" ? (
        <>
          <label>Full-body photo (front-facing, head to feet visible)
            <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} /></label>
          {file && <img className="preview" src={URL.createObjectURL(file)} alt="preview" />}
          <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", color: "#166534", padding: "10px 14px", borderRadius: "8px", margin: "8px 0", fontSize: "13px", lineHeight: "1.4" }}>
            <strong>💡 Pose Tip for Clean AI Try-On:</strong> Stand straight facing the camera with your arms naturally by your sides. Avoid crossing your arms or placing hands over your chest, as overlapping limbs cause extra hand/arm artifacts in AI models.
          </div>
          <p className="hint">Bust and hip are estimated from one 2-D photo and are approximate. You can override anything below.</p>
        </>
      ) : null}
      <div className="grid3">
        <label>Bust / chest (cm){mode === "manual" ? " *" : ""}<input type="number" value={f.bust} onChange={set("bust")} /></label>
        <label>Waist (cm)<input type="number" value={f.waist} onChange={set("waist")} /></label>
        <label>Hip (cm){mode === "manual" ? " *" : ""}<input type="number" value={f.hip} onChange={set("hip")} /></label>
      </div>
      <div className="grid3">
        <label>Body shape<select value={f.shape} onChange={set("shape")}><option value="">{mode === "photo" ? "Auto-detect" : "From measurements"}</option>{shapes.map((s) => <option key={s}>{s}</option>)}</select></label>
        <label>Undertone<select value={f.undertone} onChange={set("undertone")}><option value="">{mode === "photo" ? "Auto-detect" : "Neutral"}</option><option value="warm">warm</option><option value="cool">cool</option><option value="neutral">neutral</option></select></label>
        <label>Skin depth<select value={f.depth} onChange={set("depth")}><option value="">{mode === "photo" ? "Auto-detect" : "Medium"}</option><option>Fair</option><option>Medium</option><option>Deep</option></select></label>
      </div>
      <h3>Preferences (optional)</h3>
      <div className="chips">{(cfg?.styles || []).map((s) => <button key={s} className={p.styles.includes(s) ? "chip on" : "chip"} onClick={() => toggle(s)}>{s}</button>)}</div>
      <label>Occasion<select value={p.occasion} onChange={(e) => setP({ ...p, occasion: e.target.value })}><option value="">Any</option>{(cfg?.occasions || []).map((o) => <option key={o}>{o}</option>)}</select></label>
      {err && <div className="banner err">{err}</div>}
      <button className="primary" disabled={busy || !apiReady} onClick={submit}>
        {busy ? "Analysing…" : apiReady ? "Analyze my profile" : apiConnecting ? "Connecting to backend…" : "Backend unavailable"}
      </button>
    </section>
  );
}
