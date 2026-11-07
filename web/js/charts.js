/* Tiny canvas charts. No library, no build step. */
const DPR = Math.min(2, window.devicePixelRatio || 1);

function setup(canvas, height) {
  if (!canvas) return null;
  const w = canvas.clientWidth || canvas.parentElement?.clientWidth || 320;
  const h = height || canvas.clientHeight || 160;
  canvas.width = w * DPR;
  canvas.height = h * DPR;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function line(ctx, pts, color, width = 1.4, dash = []) {
  if (!pts.length) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  ctx.restore();
}

function dot(ctx, x, y, r, fill, stroke) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
}

export function drawSpark(canvas, points, color = "#ff3b2f") {
  const s = setup(canvas, 44);
  if (!s || !points?.length) return;
  const { ctx, w, h } = s;
  const vals = points.map((p) => p.creds ?? p);
  const max = Math.max(1, ...vals);
  const step = w / Math.max(1, vals.length - 1);
  const xy = vals.map((v, i) => [i * step, h - 4 - (v / max) * (h - 10)]);
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, color + "44");
  g.addColorStop(1, color + "00");
  ctx.beginPath();
  xy.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = g; ctx.fill();
  line(ctx, xy, color, 1.2);
  const last = xy[xy.length - 1];
  dot(ctx, last[0], last[1], 2, color, "#080b0c");
}

export function drawFrontier(canvas, curve, current = null) {
  const s = setup(canvas, 220);
  if (!s || !curve?.length) return;
  const { ctx, w, h } = s;
  const pad = { l: 32, r: 12, t: 14, b: 24 };
  const maxD = Math.max(...curve.map((p) => p.disruption), 0.2);
  const maxC = Math.max(...curve.map((p) => p.coverage), 0.4);
  const X = (d) => pad.l + (d / maxD) * (w - pad.l - pad.r);
  const Y = (c) => h - pad.b - (c / (maxC * 1.1)) * (h - pad.t - pad.b);
  // grid
  ctx.strokeStyle = "rgba(232,227,217,0.07)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * (h - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }
  const pts = curve.map((p) => [X(p.disruption), Y(p.coverage)]);
  const g = ctx.createLinearGradient(0, pad.t, 0, h);
  g.addColorStop(0, "rgba(255,59,47,0.18)");
  g.addColorStop(1, "rgba(255,59,47,0)");
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.lineTo(pts[pts.length - 1][0], h - pad.b);
  ctx.lineTo(pts[0][0], h - pad.b);
  ctx.closePath(); ctx.fillStyle = g; ctx.fill();
  line(ctx, pts, "#ff3b2f", 1.8);
  curve.forEach((p, i) => {
    const x = X(p.disruption), y = Y(p.coverage);
    dot(ctx, x, y, 2.6, "#0a0d0e", "#ff3b2f");
    ctx.fillStyle = "#666b65";
    ctx.font = "7px 'JetBrains Mono', monospace";
    ctx.fillText(`${Math.round(p.coverage * 100)}%`, x - 8, y - 7);
  });
  if (current != null) {
    const x = X(current);
    ctx.strokeStyle = "rgba(76,227,210,0.5)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = "#4a4f4b";
  ctx.font = "7px 'JetBrains Mono', monospace";
  ctx.fillText("DISRUPTION →", pad.l, h - 8);
  ctx.save();
  ctx.translate(9, h / 2 + 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("COVERAGE", 0, 0);
  ctx.restore();
}

export function drawLossCurve(canvas, losses) {
  const s = setup(canvas, 180);
  if (!s || !losses?.length) return;
  const { ctx, w, h } = s;
  const pad = { l: 28, r: 10, t: 12, b: 20 };
  const max = Math.max(...losses), min = Math.min(...losses);
  const X = (i) => pad.l + (i / Math.max(1, losses.length - 1)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - ((v - min) / Math.max(1e-6, max - min)) * (h - pad.t - pad.b);
  ctx.strokeStyle = "rgba(232,227,217,0.06)";
  for (let i = 0; i <= 3; i++) {
    const y = pad.t + (i / 3) * (h - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }
  line(ctx, losses.map((v, i) => [X(i), Y(v)]), "#4ce3d2", 1.5);
  ctx.fillStyle = "#4a4f4b";
  ctx.font = "7px 'JetBrains Mono', monospace";
  ctx.fillText(max.toFixed(2), 4, pad.t + 4);
  ctx.fillText(min.toFixed(2), 4, h - pad.b);
}

export function drawCalibration(canvas, bins) {
  const s = setup(canvas, 180);
  if (!s || !bins?.length) return;
  const { ctx, w, h } = s;
  const pad = { l: 30, r: 12, t: 12, b: 22 };
  const X = (v) => pad.l + v * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - v * (h - pad.t - pad.b);
  ctx.strokeStyle = "rgba(232,227,217,0.16)";
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke();
  ctx.setLineDash([]);
  const pts = bins.filter((b) => b.pred != null).map((b) => [X(b.pred), Y(b.actual)]);
  line(ctx, pts, "#a78bfa", 1.4);
  bins.filter((b) => b.pred != null).forEach((b, i) => {
    dot(ctx, X(b.pred), Y(b.actual), 2.4 + Math.min(3, Math.sqrt(b.n) / 4),
        "#a78bfa", "#080b0c");
  });
  ctx.fillStyle = "#4a4f4b";
  ctx.font = "7px 'JetBrains Mono', monospace";
  ctx.fillText("PREDICTED →", pad.l, h - 7);
  ctx.fillText("1.0", 6, pad.t + 4);
}

export function drawLearning(canvas, series, height = 200) {
  const s = setup(canvas, height);
  if (!s) return;
  const { ctx, w, h } = s;
  const pad = { l: 30, r: 12, t: 14, b: 22 };
  const n = Math.max(1, series?.length - 1);
  const X = (i) => pad.l + (i / n) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - v * (h - pad.t - pad.b) * 0.92;
  ctx.strokeStyle = "rgba(232,227,217,0.07)";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * (h - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillStyle = "#444945";
    ctx.font = "7px 'JetBrains Mono', monospace";
    ctx.fillText(`${Math.round((1 - i / 4) * 100)}%`, 4, y + 2);
  }
  if (!series?.length) {
    ctx.fillStyle = "#444945";
    ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.fillText("NO REPLAY YET", pad.l + 10, h / 2);
    return;
  }
  const warm = series.map((s, i) => [X(i), Y(s.warmCoverage)]);
  const cold = series.map((s, i) => [X(i), Y(s.coldCoverage)]);
  line(ctx, cold, "#5f6660", 1.2, [4, 3]);
  line(ctx, warm, "#4ce3d2", 1.8);
  warm.forEach(([x, y], i) => dot(ctx, x, y, 2.6, "#080b0c", "#4ce3d2"));
  ctx.fillStyle = "#4a4f4b";
  ctx.font = "7px 'JetBrains Mono', monospace";
  ctx.fillText("EPISODE →", pad.l, h - 6);
}

export function drawEval(canvas, series) {
  const s = setup(canvas, 210);
  if (!s) return;
  const { ctx, w, h } = s;
  const pad = { l: 30, r: 12, t: 14, b: 24 };
  if (!series?.length) {
    ctx.fillStyle = "#444945"; ctx.font = "8px 'JetBrains Mono', monospace";
    ctx.fillText("RUN THE REPLAY", pad.l + 10, h / 2); return;
  }
  const n = series.length;
  const bw = (w - pad.l - pad.r) / (n * 3 + n * 0.8);
  const Y = (v) => h - pad.b - v * (h - pad.t - pad.b) * 0.92;
  ctx.strokeStyle = "rgba(232,227,217,0.07)";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (i / 4) * (h - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }
  series.forEach((s, i) => {
    const base = pad.l + i * (bw * 4.2) + bw * 0.5;
    [[s.naiveCoverage, "#ffb020"], [s.coldCoverage, "#4a504c"], [s.warmCoverage, "#4ce3d2"]]
      .forEach(([v, c], k) => {
        const x = base + k * (bw + 1.5);
        const y = Y(v);
        ctx.fillStyle = c;
        ctx.globalAlpha = k === 2 ? 1 : 0.72;
        ctx.fillRect(x, y, bw, h - pad.b - y);
        ctx.globalAlpha = 1;
      });
    ctx.fillStyle = "#444945";
    ctx.font = "7px 'JetBrains Mono', monospace";
    ctx.fillText(String(i + 1).padStart(2, "0"), base + bw * 1.4, h - 8);
  });
  const legend = [["naive", "#ffb020"], ["cold", "#4a504c"], ["warm", "#4ce3d2"]];
  legend.forEach(([label, color], i) => {
    const x = w - 120 + i * 40;
    ctx.fillStyle = color; ctx.fillRect(x, 2, 6, 6);
    ctx.fillStyle = "#5c615c"; ctx.font = "6.5px 'JetBrains Mono', monospace";
    ctx.fillText(label, x + 9, 8);
  });
}
