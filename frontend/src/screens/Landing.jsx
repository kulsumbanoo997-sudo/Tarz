import React from "react";

export default function Landing({ go, profileId }) {
  return (
    <section className="hero">
      <h1>Know if it suits you <em>before</em> you buy.</h1>
      <p>Upload a photo or enter your measurements. TARZ estimates your body shape, skin tone and size, ranks clothes that suit you, shows an AI try-on, and an honest stylist explains why.</p>
      <div className="cta">
        <button className="primary" onClick={() => go("analyze")}>Start analysis</button>
        {profileId && <button onClick={() => go("catalog")}>Continue with my profile</button>}
      </div>
      <ol className="steps">
        <li><b>1 · Analyze</b><span>Photo or measurements → body profile</span></li>
        <li><b>2 · Catalog</b><span>Ranked by size, shape, tone, style, occasion</span></li>
        <li><b>3 · Try-On</b><span>See the garment on your photo</span></li>
        <li><b>4 · Stylist</b><span>Honest, explainable feedback</span></li>
      </ol>
    </section>
  );
}
