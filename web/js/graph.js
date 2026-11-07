/* 2.5D identity-lineage renderer. Canvas 2D, no dependency. */
import { relColor, esc } from "./api.js";

const KIND_STYLE = {
  credential: { r: 4.1, color: "#ff5b4d", shape: "circle" },
  workload: { r: 3.4, color: "#ffb020", shape: "diamond" },
  human: { r: 2.1, color: "#5f6660", shape: "circle" },
  repo: { r: 3.0, color: "#6ea8ff", shape: "square" },
  pipeline: { r: 3.2, color: "#a78bfa", shape: "triangle" },
  role: { r: 3.6, color: "#8adf6b", shape: "hex" },
  datastore: { r: 4.6, color: "#e9e4da", shape: "critical" },
  ai_endpoint: { r: 3.2, color: "#ff4fa3", shape: "diamond" },
  idp: { r: 3.6, color: "#d6c28a", shape: "hex" },
};

export class GraphRenderer {
  constructor(canvas, tip, onSelect) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tip = tip;
    this.onSelect = onSelect;
    this.nodes = [];
    this.edges = [];
    this.byId = new Map();
    this.rotY = 0.42;
    this.rotX = -0.18;
    this.zoom = 1;
    this.autoOrbit = true;
    this.clock = 0;
    this.hover = null;
    this.selected = null;
    this.state = new Map();
    this.pathEdges = new Set();
    this.pathNodes = new Set();
    this.recalled = new Set();
    this.dragging = false;
    this.moved = 0;
    this.last = { x: 0, y: 0 };
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.particles = [];
    this._bind();
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._loop();
  }

  setData(graph) {
    this.graph = graph;
    this.nodes = graph.nodes.map((n) => ({ ...n, pos: n.pos || [0, 0, 0] }));
    this.byId = new Map(this.nodes.map((n) => [n.id, n]));
    this.edges = graph.edges.map((e) => ({ ...e, a: this.byId.get(e.src), b: this.byId.get(e.dst) }))
      .filter((e) => e.a && e.b);
    this._fit();
  }

  setIncident(inc) {
    this.incident = inc;
    this.state = new Map();
    if (inc) {
      for (const s of inc.sim.nodeStates) this.state.set(s.id, s);
      this.pathEdges = new Set();
      this.pathNodes = new Set();
      const top = inc.sim.paths.slice(0, 5);
      for (const p of top) {
        p.edges.forEach((e) => this.pathEdges.add(e));
        p.nodes.forEach((n) => this.pathNodes.add(n));
      }
      this.p0 = inc.patientZero;
      this.clock = inc.sim.horizon;
      this.particles = [];
      for (const e of this.pathEdges) {
        const edge = this.edges.find((x) => x.id === e);
        if (edge) this.particles.push({ edge, off: Math.random() });
      }
    } else {
      this.p0 = null;
      this.pathEdges.clear();
      this.pathNodes.clear();
      this.clock = 0;
    }
  }

  setRecalled(ids = []) { this.recalled = new Set(ids); }
  setClock(t) { this.clock = t; }
  setZoom(z) { this.zoom = Math.max(0.45, Math.min(2.6, z)); }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.w = rect.width;
    this.h = rect.height;
    this.canvas.width = Math.max(1, rect.width * this.dpr);
    this.canvas.height = Math.max(1, rect.height * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this._fit();
  }

  _fit() {
    if (!this.nodes.length) return;
    const xs = this.nodes.map((n) => n.pos[0]);
    const zs = this.nodes.map((n) => n.pos[2]);
    const span = Math.max(
      Math.max(...xs) - Math.min(...xs),
      Math.max(...zs) - Math.min(...zs), 0.5);
    this.base = Math.min(this.w, this.h) * 0.34 / span * 1.35;
  }

  _project(node) {
    const [x, y, z] = node.pos;
    const cy = Math.cos(this.rotY), sy = Math.sin(this.rotY);
    const cx = Math.cos(this.rotX), sx = Math.sin(this.rotX);
    let x1 = x * cy - z * sy;
    let z1 = x * sy + z * cy;
    const y1 = y * cx - z1 * sx;
    const z2 = y * sx + z1 * cx;
    const persp = 1 / (1 + z2 * 0.34);
    const scale = this.base * this.zoom * persp;
    return {
      x: this.w / 2 + x1 * scale,
      y: this.h / 2 + y1 * scale * 0.92,
      z: z2,
      s: scale,
    };
  }

  _nodeState(id) {
    if (!this.incident) return { infected: false, p: 0, t: 0 };
    if (id === this.p0) return { infected: true, p: 1, t: 0, p0: true };
    const s = this.state.get(id);
    if (!s) return { infected: false, p: 0, t: 0 };
    return { ...s, infected: s.t > 0 && s.t <= this.clock && s.p > 0.004 };
  }

  _bind() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      this.dragging = true; this.moved = 0;
      this.last = { x: e.clientX, y: e.clientY };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      if (this.dragging) {
        const dx = e.clientX - this.last.x, dy = e.clientY - this.last.y;
        this.moved += Math.abs(dx) + Math.abs(dy);
        this.rotY += dx * 0.006;
        this.rotX = Math.max(-1.1, Math.min(0.75, this.rotX + dy * 0.004));
        this.last = { x: e.clientX, y: e.clientY };
        this.tip.classList.remove("on");
        return;
      }
      const hit = this._hit(mx, my);
      this.hover = hit?.node || null;
      if (hit) {
        const { node } = hit;
        const st = this._nodeState(node.id);
        this.tip.innerHTML = `<b>${esc(node.label)}</b>
          <span>${String(node.kind).toUpperCase()} · ${esc(node.detail || node.service || "")}</span>
          <span class="tip-row"><em>reach probability</em><b>${(st.p * 100).toFixed(0)}%</b></span>
          <span class="tip-row"><em>time to reach</em><b>${st.t ? Math.round(st.t) + "m" : "—"}</b></span>
          <span class="tip-row"><em>critical</em><b>${node.critical ? "YES" : "no"}</b></span>`;
        this.tip.style.left = `${Math.min(this.w - 220, mx + 14)}px`;
        this.tip.style.top = `${Math.min(this.h - 100, my + 14)}px`;
        this.tip.classList.add("on");
      } else {
        this.tip.classList.remove("on");
      }
    });
    const up = (e) => {
      if (this.dragging && this.moved < 5) {
        const rect = c.getBoundingClientRect();
        const hit = this._hit(e.clientX - rect.left, e.clientY - rect.top);
        if (hit) {
          this.selected = hit.node;
          this.onSelect?.(hit.node, hit.state);
        }
      }
      this.dragging = false;
    };
    c.addEventListener("pointerup", up);
    c.addEventListener("pointerleave", () => { this.dragging = false; this.tip.classList.remove("on"); });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.setZoom(this.zoom * (e.deltaY > 0 ? 0.92 : 1.08));
    }, { passive: false });
  }

  _hit(mx, my) {
    let best = null;
    for (const node of this.nodes) {
      const p = this._project(node);
      const st = this._nodeState(node.id);
      const style = KIND_STYLE[node.kind] || KIND_STYLE.credential;
      const r = Math.max(7, style.r * Math.max(0.7, p.s / 260) * 1.4);
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < r + 3 && (!best || p.z < best.p.z)) best = { node, p, state: st };
    }
    return best;
  }

  _loop() {
    if (this.autoOrbit && !this.dragging) this.rotY += 0.0011;
    this._draw();
    requestAnimationFrame(() => this._loop());
  }

  _drawShape(ctx, style, x, y, r, color, alpha = 1) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    switch (style.shape) {
      case "square":
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
        break;
      case "critical":
        ctx.translate(x, y); ctx.rotate(Math.PI / 4);
        ctx.fillRect(-r * 0.9, -r * 0.9, r * 1.8, r * 1.8);
        ctx.strokeStyle = "#080b0c"; ctx.lineWidth = 1.4;
        ctx.strokeRect(-r * 0.9, -r * 0.9, r * 1.8, r * 1.8);
        break;
      case "diamond":
        ctx.beginPath();
        ctx.moveTo(x, y - r * 1.25); ctx.lineTo(x + r, y);
        ctx.lineTo(x, y + r * 1.25); ctx.lineTo(x - r, y);
        ctx.closePath(); ctx.fill();
        break;
      case "triangle":
        ctx.beginPath();
        ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r, y + r * 0.85);
        ctx.lineTo(x - r, y + r * 0.85); ctx.closePath(); ctx.fill();
        break;
      case "hex":
        ctx.beginPath();
        for (let i = 0; i < 6; i++) {
          const a = Math.PI / 3 * i - Math.PI / 6;
          const px = x + Math.cos(a) * r, py = y + Math.sin(a) * r;
          i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.closePath(); ctx.fill();
        break;
      default:
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
  }

  _draw() {
    const { ctx, w, h } = this;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);

    // floor rings
    ctx.save();
    ctx.strokeStyle = "rgba(232,227,217,0.045)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath();
      ctx.ellipse(w / 2, h * 0.56, (this.base * this.zoom * i) / 2.4,
        (this.base * this.zoom * i) / 8.5, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();

    // edges
    const projected = new Map();
    for (const n of this.nodes) projected.set(n.id, this._project(n));
    const edgeOrder = [...this.edges].sort((a, b) => {
      const za = (projected.get(a.src).z + projected.get(a.dst).z) / 2;
      const zb = (projected.get(b.src).z + projected.get(b.dst).z) / 2;
      return zb - za;
    });

    for (const e of edgeOrder) {
      const pa = projected.get(e.src), pb = projected.get(e.dst);
      if (!pa || !pb) continue;
      const both = this._nodeState(e.src).infected && this._nodeState(e.dst).infected;
      const onPath = this.pathEdges.has(e.id);
      const isHover = this.hover && (e.src === this.hover.id || e.dst === this.hover.id);
      let alpha = 0.055 + e.conf * 0.16;
      if (both) alpha += 0.10;
      if (onPath) alpha = 0.55;
      if (isHover) alpha = 0.5;
      const color = onPath ? "#ff3b2f" : relColor[e.rel] || "#6f766f";
      const mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
      const dx = pb.x - pa.x, dy = pb.y - pa.y;
      const curve = 0.10 + Math.min(0.16, Math.hypot(dx, dy) / (w * 1.4));
      const cx = mx - dy * curve, cy = my + dx * curve;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = onPath ? 1.15 : 0.65;
      if (e.source === "inferred" && !onPath) ctx.setLineDash([2, 4]);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.quadraticCurveTo(cx, cy, pb.x, pb.y);
      ctx.stroke();
      ctx.restore();

      if (onPath) {
        const p = this.particles.find((x) => x.edge.id === e.id);
        if (p) {
          const t = (performance.now() / 3400 + p.off) % 1;
          const q = 1 - t;
          const px = q * q * pa.x + 2 * q * t * cx + t * t * pb.x;
          const py = q * q * pa.y + 2 * q * t * cy + t * t * pb.y;
          ctx.save();
          ctx.globalAlpha = 0.9;
          ctx.fillStyle = "#ffb020";
          ctx.shadowColor = "#ff3b2f"; ctx.shadowBlur = 10;
          ctx.beginPath(); ctx.arc(px, py, 1.6, 0, Math.PI * 2); ctx.fill();
          ctx.restore();
        }
      }
    }

    // nodes sorted back to front
    const sorted = [...this.nodes].sort((a, b) => projected.get(b.id).z - projected.get(a.id).z);
    for (const node of sorted) {
      const p = projected.get(node.id);
      const st = this._nodeState(node.id);
      const style = KIND_STYLE[node.kind] || KIND_STYLE.credential;
      const baseR = style.r * Math.max(0.62, Math.min(1.5, p.s / 260));
      let color = style.color;
      let alpha = 0.58;
      if (node.kind === "credential") alpha = 0.9;
      if (node.critical) alpha = 1;
      if (st.infected) {
        const heat = Math.max(0, 1 - this.clock / Math.max(1, this.incident?.sim.horizon || 90));
        color = st.p0 ? "#ff3b2f" : (this.clock < 30 ? "#ff7a45" : "#ffb020");
        alpha = 1;
      }
      if (this.hover && this.hover.id !== node.id
          && !this.edges.some((e) => (e.src === this.hover.id && e.dst === node.id)
            || (e.dst === this.hover.id && e.src === node.id))) {
        alpha *= 0.55;
      }
      if (st.infected) {
        ctx.save();
        ctx.globalAlpha = st.p0 ? 0.28 : 0.16;
        const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, baseR * 8);
        grd.addColorStop(0, st.p0 ? "#ff3b2f" : "#ffb020");
        grd.addColorStop(1, "transparent");
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.arc(p.x, p.y, baseR * 8, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
      }
      if (st.p0) {
        const pulse = 1 + 0.35 * Math.sin(performance.now() / 320);
        ctx.save();
        ctx.globalAlpha = 0.5;
        ctx.strokeStyle = "#ff3b2f"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(p.x, p.y, baseR * 4.2 * pulse, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 0.22;
        ctx.beginPath(); ctx.arc(p.x, p.y, baseR * 6.4 * pulse, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      }
      if (this.recalled.has(node.id)) {
        ctx.save();
        ctx.globalAlpha = 0.7;
        ctx.strokeStyle = "#4ce3d2"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(p.x, p.y, baseR + 4, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      }
      ctx.save();
      if (st.infected) {
        ctx.shadowColor = color; ctx.shadowBlur = node.critical ? 22 : 12;
      }
      this._drawShape(ctx, style, p.x, p.y, baseR, color, alpha);
      ctx.restore();

      const showLabel = node.critical || node.id === this.p0 || this.hover?.id === node.id
        || (this.pathNodes.has(node.id) && this.zoom > 0.85)
        || (st.infected && this.zoom > 1.25);
      if (showLabel) {
        ctx.save();
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = node.critical ? "#e9e4da" : (st.infected ? "#ffcf87" : "#9aa097");
        ctx.font = `${node.critical ? 7.5 : 6.5}px "JetBrains Mono", monospace`;
        ctx.fillText(node.label.slice(0, 24), p.x + baseR + 5, p.y + 2.5);
        ctx.restore();
      }
    }

    // vignette + crosshair
    ctx.save();
    const vg = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.25,
      w / 2, h / 2, Math.max(w, h) * 0.72);
    vg.addColorStop(0, "rgba(0,0,0,0)");
    vg.addColorStop(1, "rgba(0,0,0,0.55)");
    ctx.fillStyle = vg; ctx.fillRect(0, 0, w, h);
    ctx.restore();
  }
}

export class MemoryGraph {
  constructor(canvas, onSelect) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onSelect = onSelect;
    this.records = [];
    this.nodes = [];
    this.links = [];
    this.hover = null;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.addEventListener("pointermove", (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      this.hover = this.nodes.find((n) => Math.hypot(n.x - x, n.y - y) < n.r + 4) || null;
      canvas.style.cursor = this.hover ? "pointer" : "default";
    });
    canvas.addEventListener("pointerdown", (e) => {
      if (this.hover) this.onSelect?.(this.hover.record);
    });
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.w = r.width; this.h = r.height;
    this.canvas.width = Math.max(1, r.width * this.dpr);
    this.canvas.height = Math.max(1, r.height * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.layout();
    this.draw();
  }

  setRecords(records) {
    this.records = records || [];
    this.layout();
    this.draw();
  }

  layout() {
    const recs = this.records.slice(-46);
    const cx = this.w / 2, cy = this.h / 2;
    const groups = { episode: [], fact: [], directive: [] };
    recs.forEach((r) => (groups[r.type] || groups.fact).push(r));
    this.nodes = [];
    const rings = { episode: 0.28, fact: 0.55, directive: 0.78 };
    for (const [type, arr] of Object.entries(groups)) {
      arr.forEach((rec, i) => {
        const a = (i / Math.max(1, arr.length)) * Math.PI * 2
          + (type === "fact" ? 0.6 : type === "directive" ? 1.2 : 0);
        const rad = Math.min(this.w, this.h) * (rings[type] / 2);
        this.nodes.push({
          x: cx + Math.cos(a) * rad * (0.9 + 0.1 * Math.sin(i * 7)),
          y: cy + Math.sin(a) * rad * (0.86 + 0.1 * Math.cos(i * 5)),
          r: type === "episode" ? 4.2 : type === "directive" ? 3.6 : 2.6,
          type, record: rec,
        });
      });
    }
    this.links = [];
    for (let i = 0; i < this.nodes.length; i++) {
      for (let j = i + 1; j < this.nodes.length; j++) {
        const a = this.nodes[i], b = this.nodes[j];
        const ea = new Set((a.record.meta?.entities || a.record.entities || []));
        const eb = new Set((b.record.meta?.entities || b.record.entities || []));
        let shared = 0;
        ea.forEach((x) => { if (eb.has(x) && x.length > 3) shared++; });
        if (shared > 0) this.links.push([a, b, shared]);
      }
    }
    this.links = this.links.slice(0, 120);
  }

  draw() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.strokeStyle = "rgba(76,227,210,0.06)";
    for (let i = 1; i <= 3; i++) {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, Math.min(w, h) * (i / 7), 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
    for (const [a, b, shared] of this.links) {
      ctx.save();
      ctx.globalAlpha = Math.min(0.25, 0.05 + shared * 0.04);
      ctx.strokeStyle = "#4ce3d2";
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.restore();
    }
    for (const n of this.nodes) {
      const color = n.type === "episode" ? "#4ce3d2" : n.type === "directive" ? "#ff3b2f" : "#e9e4da";
      ctx.save();
      ctx.globalAlpha = this.hover === n ? 1 : 0.75;
      ctx.fillStyle = color;
      if (this.hover === n) { ctx.shadowColor = color; ctx.shadowBlur = 12; }
      ctx.beginPath();
      if (n.type === "directive") {
        ctx.moveTo(n.x, n.y - n.r); ctx.lineTo(n.x + n.r, n.y);
        ctx.lineTo(n.x, n.y + n.r); ctx.lineTo(n.x - n.r, n.y);
        ctx.closePath(); ctx.fill();
      } else {
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2); ctx.fill();
      }
      ctx.restore();
      if (n.type === "episode" || this.hover === n) {
        ctx.save();
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = "#6c726c";
        ctx.font = "6px 'JetBrains Mono', monospace";
        const text = (n.record.text || "").slice(0, 26);
        ctx.fillText(text, n.x + n.r + 4, n.y + 2);
        ctx.restore();
      }
    }
    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = "#4a4f4b";
    ctx.font = "7px 'JetBrains Mono', monospace";
    ctx.fillText(`${this.nodes.length} MEMORIES · ${this.links.length} ENTITY LINKS`, 8, h - 8);
    ctx.restore();
  }
}
