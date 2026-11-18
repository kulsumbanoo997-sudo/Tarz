export const API = "";
export const img = (u) => (!u ? "" : u.startsWith("http") ? u : API + u);

const retryDelays = [3000, 7000, 15000, 25000];

const isTransient = (error) =>
  error instanceof TypeError || [502, 503, 504].includes(error && error.status);

async function retryTransient(request) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await request();
    } catch (error) {
      if (!isTransient(error) || attempt >= retryDelays.length) throw error;
      await new Promise((resolve) => setTimeout(resolve, retryDelays[attempt]));
    }
  }
}

async function handle(r) {
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    const error = new Error(msg || `HTTP ${r.status}`);
    error.status = r.status;
    throw error;
  }
  return r.json();
}
const json = (path, body) => fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(handle);

export const getConfig = () => retryTransient(() => fetch(API + "/api/config").then(handle));
export const analyze = (formData) => fetch(API + "/api/analyze", { method: "POST", body: formData }).then(handle);
export const recommend = (body) => retryTransient(() => json("/api/recommend", body));
export const askStylist = (body) => json("/api/stylist", body);
export const tryOn = (profile_id, product_id) => {
  const f = new FormData(); f.append("profile_id", profile_id); f.append("product_id", product_id);
  return fetch(API + "/api/tryon", { method: "POST", body: f }).then(handle);
};
export const deleteProfile = (id) => fetch(API + `/api/profile/${id}`, { method: "DELETE" }).then(handle);
