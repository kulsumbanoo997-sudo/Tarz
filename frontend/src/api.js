export const API = "";
export const img = (u) => (!u ? "" : u.startsWith("http") ? u : API + u);

async function handle(r) {
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg || `HTTP ${r.status}`);
  }
  return r.json();
}
const json = (path, body) => fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle);

export const getConfig = () => fetch(API + "/api/config").then(handle);
export const analyze = (formData) => fetch(API + "/api/analyze", { method: "POST", body: formData }).then(handle);
export const recommend = (body) => json("/api/recommend", body);
export const askStylist = (body) => json("/api/stylist", body);
export const tryOn = (profile_id, product_id) => {
  const f = new FormData(); f.append("profile_id", profile_id); f.append("product_id", product_id);
  return fetch(API + "/api/tryon", { method: "POST", body: f }).then(handle);
};
export const deleteProfile = (id) => fetch(API + `/api/profile/${id}`, { method: "DELETE" }).then(handle);
