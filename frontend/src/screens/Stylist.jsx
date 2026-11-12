import React, { useEffect, useState } from "react";
import { askStylist, img } from "../api.js";

const VCLASS = { suitable: "good", "partly suitable": "mid", unsuitable: "bad" };

export default function Stylist({ go, profileId, prefs, product }) {
  const [out, setOut] = useState(null);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const ask = async (question = "") => {
    setBusy(true); setErr("");
    try { setOut(await askStylist({ profile_id: profileId, product_id: product.id, preferences: prefs, question })); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  };
  useEffect(() => { if (product) { setOut(null); ask(); } /* eslint-disable-next-line */ }, [product?.id]);

  if (!product) return <section className="card narrow"><p>Pick a garment from the catalog first.</p><button className="primary" onClick={() => go("catalog")}>Open catalog</button></section>;
  return (
    <section className="card">
      <h2>AI Stylist</h2>
      <div className="stylist">
        <figure><img src={img(product.image)} alt={product.name} /><figcaption>{product.name}</figcaption></figure>
        <div>
          {busy && !out && <p className="hint">The stylist is thinking…</p>}
          {err && <div className="banner err">{err}</div>}
          {out && <>
            <div className={"verdict " + VCLASS[out.verdict]}>{out.verdict} · score {Math.round(out.score.total * 100)}%</div>
            <h3>{out.headline}</h3>
            <ul>{out.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
            {out.tips.map((t, i) => <p key={i} className="hint">💡 {t}</p>)}
            {out.answer && <div className="answer"><b>Your question:</b> {out.answer}</div>}
            {out.alternatives.length > 0 && <>
              <h4>Alternatives</h4>
              <div className="alts">{out.alternatives.map((a) => a.product && (
                <div key={a.id} className="alt" onClick={() => go("stylist", a.product)}>
                  <img src={img(a.product.image)} alt="" /><div><b>{a.product.name}</b><small>{a.why}</small></div></div>))}</div></>}
            <small className="hint">Explained by: {out.model}. The verdict comes from the score, not from the AI.</small>
          </>}
          <div className="ask"><input placeholder="Ask a follow-up, e.g. “Is this OK for a wedding?”" value={q} onChange={(e) => setQ(e.target.value)} />
            <button disabled={busy || !q.trim()} onClick={() => ask(q)}>Ask</button></div>
        </div>
      </div>
      <div className="row"><button onClick={() => go("tryon", product)}>Try it on</button><button onClick={() => go("catalog")}>Back to catalog</button></div>
    </section>
  );
}
