import React, { useEffect, useState } from "react";
import Landing from "./screens/Landing.jsx";
import Analyze from "./screens/Analyze.jsx";
import Catalog from "./screens/Catalog.jsx";
import TryOn from "./screens/TryOn.jsx";
import Stylist from "./screens/Stylist.jsx";
import { getConfig } from "./api.js";

const STEPS = [["landing", "Home"], ["analyze", "Analyze"], ["catalog", "Catalog"], ["tryon", "Try-On"], ["stylist", "Stylist"]];
const KEY = "tarz_state_v1";

export default function App() {
  const saved = (() => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } })();
  const [screen, setScreen] = useState(saved.profileId ? "catalog" : "landing");
  const [profileId, setProfileId] = useState(saved.profileId || null);
  const [profile, setProfile] = useState(saved.profile || null);
  const [hasPhoto, setHasPhoto] = useState(!!saved.hasPhoto);
  const [prefs, setPrefs] = useState(saved.prefs || { styles: [], occasion: "" });
  const [product, setProduct] = useState(null);
  const [cfg, setCfg] = useState(null);
  const [cfgErr, setCfgErr] = useState("");

  useEffect(() => { getConfig().then(setCfg).catch(() => setCfgErr("Cannot reach the TARZ API. Start the backend (see README).")); }, []);
  useEffect(() => { localStorage.setItem(KEY, JSON.stringify({ profileId, profile, hasPhoto, prefs })); }, [profileId, profile, hasPhoto, prefs]);

  const go = (s, p) => { if (p) setProduct(p); setScreen(s); window.scrollTo(0, 0); };
  const onProfile = (r, p) => { setProfileId(r.profile_id); setProfile(r.profile); setHasPhoto(r.has_photo); setPrefs(p); go("catalog"); };
  const reset = () => { setProfileId(null); setProfile(null); setHasPhoto(false); setProduct(null); go("landing"); };
  const ctx = { go, profileId, profile, hasPhoto, prefs, setPrefs, product, cfg, reset, onProfile };

  return (
    <div className="app">
      <header className="top">
        <div className="brand" onClick={() => go("landing")}>TARZ<span> · try before you buy</span></div>
        <nav>{STEPS.map(([k, l]) => (
          <button key={k} className={screen === k ? "on" : ""} onClick={() => go(k)}
            disabled={k !== "landing" && k !== "analyze" && !profileId}>{l}</button>))}</nav>
      </header>
      {cfgErr && <div className="banner err">{cfgErr}</div>}
      <main>
        {screen === "landing" && <Landing {...ctx} />}
        {screen === "analyze" && <Analyze {...ctx} />}
        {screen === "catalog" && <Catalog {...ctx} />}
        {screen === "tryon" && <TryOn {...ctx} />}
        {screen === "stylist" && <Stylist {...ctx} />}
      </main>
      <footer>AI suggestions are approximate guides, not guarantees of fit. Photos are stored on the TARZ server until you delete them.</footer>
    </div>
  );
}
