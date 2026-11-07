export async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(data.error || data.detail || `HTTP ${res.status}`);
  return data;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, cls, html) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

export const fmt = {
  int(n) { return (n ?? 0).toLocaleString("en-US"); },
  usd(n) {
    if (n == null) return "—";
    if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
    if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e4) return `$${Math.round(n / 1000)}K`;
    return `$${fmt.int(Math.round(n))}`;
  },
  usd0(n) {
    if (n == null) return "—";
    if (n >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `$${Math.round(n / 1e3)}K`;
    return `$${Math.round(n)}`;
  },
  pct(n, d = 0) { return n == null ? "—" : `${(n * 100).toFixed(d)}%`; },
  num(n, d = 1) { return n == null ? "—" : Number(n).toFixed(d); },
  mins(n) { return n == null ? "—" : `${Math.round(n)}m`; },
  time(iso) {
    if (!iso) return "—";
    try { return new Date(iso).toLocaleTimeString("en-GB", { hour12: false }); }
    catch { return "—"; }
  },
  secs(ms) { return ms == null ? "—" : `${(ms / 1000).toFixed(2)}s`; },
};

export const relColor = {
  contains_secret: "#ff5b4d",
  mounts: "#ff7a45",
  can_assume: "#ffb020",
  can_read: "#8adf6b",
  can_write: "#4ce3d2",
  can_deploy: "#a78bfa",
  can_invoke: "#ff4fa3",
  push_access: "#6ea8ff",
  member_of: "#5f6660",
  owns: "#3e433f",
  trusts: "#d6c28a",
  authenticates: "#9aa39b",
};

export const relLabel = {
  contains_secret: "contains secret",
  mounts: "mounts",
  can_assume: "can assume",
  can_read: "can read",
  can_write: "can write",
  can_deploy: "can deploy",
  can_invoke: "can invoke",
  push_access: "push access",
  member_of: "member of",
  owns: "owns",
  trusts: "trusts",
  authenticates: "authenticates",
};
