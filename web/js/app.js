import { api, $, $$, esc, fmt } from "./api.js";
import { GraphRenderer, MemoryGraph } from "./graph.js";
import {
  renderScenarios, markScenario, renderIncidents, renderIncident,
  renderMemoryStats, renderRecall, renderModel, renderEvalData,
  renderPlanDesign, renderChoreo, openModal, closeModal, log, toast,
  showLoading, updateClock, animateTimeline,
} from "./ui.js";

const state = {
  bootstrap: null,
  graph: null,
  incident: null,
  eval: null,
  scenario: null,
  budget: 0.08,
  memoryOn: true,
  tab: "outbreak",
  booted: false,
  clockTimer: null,
};

let graph;
let memGraph;

/* ------------------------------------------------------------------ boot */
const BOOT_LINES = [
  ["mounting identity lineage graph", "80 nodes · 143 evidenced edges · 5 critical assets"],
  ["loading contagion-gnn", "relation-aware GraphSAGE · numpy autograd"],
  ["loading contagion-lm", "decoder-only transformer · trained here"],
  ["opening hindsight bank", "four-channel recall · learned posteriors"],
  ["arming outbreak simulator", "monte carlo · detection hazard · attacker policy"],
  ["console ready", "the scanner found a key; this finds the outbreak"],
];

async function boot() {
  const box = $("#bootLog");
  for (const [line, detail] of BOOT_LINES) {
    const row = document.createElement("div");
    row.innerHTML = `<span class="ok">[ok]</span> ${esc(line)} <span style="color:#3e433f">${esc(detail)}</span>`;
    box.appendChild(row);
    await sleep(120);
  }
  $("#bootEnter").disabled = false;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

$("#bootEnter").onclick = async () => {
  $("#boot").classList.add("done");
  $("#app").classList.remove("is-booting");
  state.booted = true;
  log("ok", "console online · synthetic estate · destructive actions simulated");
  await runScenario(state.scenarios[0]);
};

/* --------------------------------------------------------------- data init */
async function init() {
  graph = new GraphRenderer($("#graph"), $("#graphTip"), onNodeSelect);
  memGraph = new MemoryGraph($("#memCanvas"), (rec) => {
    openModal("MEMORY RECORD", `<p style="font-family:'JetBrains Mono';font-size:10px;line-height:1.9;color:#b9bdb5">${esc(rec.text)}</p>
      <p class="evidence-note">TYPE ${esc(rec.type.toUpperCase())} · UTILITY ${rec.utility ?? "—"} · ${esc(rec.ts || "")}</p>`);
  });

  const [bootData, g] = await Promise.all([api("/api/bootstrap"), api("/api/graph?estate=meridian")]);
  state.bootstrap = bootData;
  state.scenarios = bootData.scenarios;
  graph.setData(g);
  window.__graph = g;
  $("#graphTitle").textContent =
    `${g.name.toUpperCase()} · ${g.stats.nodes} NODES · ${g.stats.edges} EDGES`;

  const sel = $("#estateSelect");
  sel.innerHTML = bootData.estates.map(
    (e) => `<option value="${e.id}">${esc(e.name)}</option>`).join("");
  sel.onchange = () => loadEstate(sel.value);

  renderScenarios(bootData.scenarios, runScenario);
  renderIncidents(bootData.incidents, openIncident);
  renderMemoryStats(bootData.memory);
  $("#modelReadout").textContent =
    `GNN ${((bootData.gnn.params || 0) / 1000).toFixed(1)}K · AP ${bootData.gnn.node?.ap ?? "—"}`;
  $("#memReadout").textContent =
    `${bootData.memory.records} RECORDS · ${(bootData.memory.mode || "local").split(" ")[0].toUpperCase()}`;
  $("#memBackend").textContent = (bootData.memory.mode || "local").toUpperCase();
  renderPlanDesign(bootData.system);

  await Promise.all([refreshMemoryStats(), refreshEval()]);
  if (bootData.campaign) renderEvalData(bootData.campaign);
  wire();
  boot();
  setInterval(updateClock, 1000);
  updateClock();
}

async function refreshEval() {
  try {
    state.eval = await api("/api/eval");
    renderModel(state.eval);
  } catch (err) { log("err", `eval unavailable: ${err.message}`); }
}

async function refreshMemoryStats() {
  try {
    const m = await api("/api/memory/stats");
    renderMemoryStats(m, m.records);
    memGraph.setRecords(m.records || []);
    $("#memBackend").textContent = (m.stats?.mode || "local").toUpperCase();
    $("#memCount").textContent = String(m.stats?.records || 0);
    $("#memReadout").textContent =
      `${m.stats?.records || 0} RECORDS · ${(m.stats?.mode || "local").split(" ")[0].toUpperCase()}`;
  } catch (err) { log("err", `memory stats: ${err.message}`); }
}

async function loadEstate(id) {
  const g = await api(`/api/graph?estate=${id}`);
  state.graph = g;
  window.__graph = g;
  graph.setData(g);
  graph.setIncident(null);
  clearIncidentView();
  $("#graphTitle").textContent =
    `${g.name.toUpperCase()} · ${g.stats.nodes} NODES · ${g.stats.edges} EDGES`;
  log("graph", `${g.name} loaded · ${g.stats.nodes} nodes · ${g.stats.edges} edges`);
}

function clearIncidentView() {
  state.incident = null;
  $("#kpiR0").textContent = "—";
  $("#kpiTtc").textContent = "—";
  $("#kpiReach").textContent = "—";
  $("#kpiLoss").textContent = "—";
  $("#kpiSpread").textContent = "—";
  $("#pathList").innerHTML = `<div class="empty-line">run a scenario to simulate the outbreak</div>`;
  $("#planList").innerHTML = `<div class="empty-line">no plan yet</div>`;
  $("#executeBtn").disabled = true;
  $("#planExecuteBtn").disabled = true;
  $("#beliefBadge").textContent = "COLD";
  $("#beliefBadge").className = "badge";
  $("#reachedCount").textContent = "0";
  $("#clockReadout").textContent = "T+00:00";
  $("#phaseLabel").textContent = "AWAITING PATIENT ZERO";
}

/* ------------------------------------------------------------------ run */
async function runScenario(scenario) {
  if (!scenario) return;
  state.scenario = scenario;
  markScenario(scenario.id);
  $("#estateSelect").value = scenario.estate;
  const stop = showLoading([
    "recalling prior outbreaks from Hindsight",
    "blending graph priors into edge posteriors",
    "propagating the outbreak across the estate",
    "optimising the ring-vaccination portfolio",
    "drafting memo · validating every number",
  ]);
  const t0 = performance.now();
  log("recall", `querying memory for ${scenario.label} · memory ${state.memoryOn ? "on" : "off"}`);
  try {
    if (state.graph?.id !== scenario.estate) {
      const g = await api(`/api/graph?estate=${scenario.estate}`);
      state.graph = g; window.__graph = g; graph.setData(g);
      $("#graphTitle").textContent =
        `${g.name.toUpperCase()} · ${g.stats.nodes} NODES · ${g.stats.edges} EDGES`;
    }
    const inc = await api("/api/run", {
      estate: scenario.estate,
      patientZero: scenario.p0,
      trials: Number($("#trialsSelect").value),
      horizon: Number($("#horizonSelect").value),
      policy: $("#policySelect").value,
      useMemory: state.memoryOn,
      budget: state.budget,
    });
    state.incident = inc;
    stop();
    setTab("outbreak");
    renderIncident(inc, graph);
    if (inc.recalled?.length) renderRecall({ results: inc.recalled, latencyMs: inc.trace.find((t) => t.step === "recall")?.ms || 0 });
    log("sim", `${inc.id} · ${inc.sim.trials.toLocaleString()} outbreaks · R₀ ${inc.sim.kpis.r0} · ${Math.round(inc.plan.coverage * 100)}% coverage`);
    log(inc.memoryUsed ? "mem" : "warn",
      inc.memoryUsed
        ? `Hindsight recalled ${inc.recalled.length} records · ${Object.keys(inc.beliefs.actionPriors).length} action priors · ${inc.beliefs.directives.length} directives`
        : "cold start · no prior outbreak consulted");
    log("narrate", `memo ${inc.memo.validator.final.accepted}/${inc.memo.validator.final.checked} claims verified · draft ${inc.memo.validator.accepted}/${inc.memo.validator.checked} grounded`);
    $("#ticker").textContent =
      `${inc.id} · ${scenario.label.toUpperCase()} · R₀ ${inc.sim.kpis.r0} · ` +
      `${Math.round(inc.plan.coverage * 100)}% CUT · ${inc.memoryUsed ? "HINDSIGHT WARM" : "COLD"}`;
    toast(`${inc.id} contained in simulation · ${(performance.now() - t0).toFixed(0)}ms`);
    await refreshAfterRun();
  } catch (err) {
    stop();
    log("err", `run failed: ${err.message}`);
    toast(`run failed: ${err.message}`, true);
  }
}

async function refreshAfterRun() {
  try {
    const [b, m] = await Promise.all([api("/api/bootstrap"), api("/api/memory/stats")]);
    state.bootstrap = b;
    renderIncidents(b.incidents, openIncident);
    renderMemoryStats(m, m.records);
    memGraph.setRecords(m.records || []);
    $("#memReadout").textContent =
      `${m.stats?.records || 0} RECORDS · ${(m.stats?.mode || "local").split(" ")[0].toUpperCase()}`;
  } catch (err) { /* non-fatal */ }
}

function openIncident(meta) {
  const inc = state.bootstrap?.incidents?.find((i) => i.id === meta.id);
  if (!inc) { toast("incident payload not in this session; re-run the scenario"); return; }
  api(`/api/incident/${meta.id}`).then((full) => {
    state.incident = full;
    renderIncident(full, graph);
    setTab("outbreak");
  }).catch((err) => toast(err.message, true));
}

/* ------------------------------------------------------------------ actions */
function wire() {
  $$(".tab").forEach((btn) => {
    btn.onclick = () => setTab(btn.dataset.tab);
  });
  $("#memoryToggle").onclick = () => {
    state.memoryOn = !state.memoryOn;
    const t = $("#memoryToggle");
    t.classList.toggle("on", state.memoryOn);
    t.querySelector(".toggle-label").textContent = state.memoryOn ? "MEMORY ON" : "MEMORY OFF";
    $("#memoryModeNote").textContent = state.memoryOn
      ? "Hindsight: local mirror · cloud when configured"
      : "memory disabled · every plan starts from zero";
    log(state.memoryOn ? "mem" : "warn", state.memoryOn
      ? "Hindsight re-enabled for the next run"
      : "Hindsight disabled · cold-start mode");
  };
  $("#executeBtn").onclick = executeFlow;
  $("#planExecuteBtn").onclick = executeFlow;
  $("#playBtn").onclick = replayTimeline;
  $("#scrub").oninput = (e) => {
    const t = Number(e.target.value);
    if (state.incident) {
      graph.setClock(t);
      updateScrub(state.incident, t);
    }
  };
  $("#orbitBtn").onclick = (e) => {
    graph.autoOrbit = !graph.autoOrbit;
    e.currentTarget.classList.toggle("on", graph.autoOrbit);
  };
  $("#resetViewBtn").onclick = () => {
    graph.rotY = 0.42; graph.rotX = -0.18; graph.zoom = 1;
  };
  $("#fitBtn").onclick = () => { graph.zoom = 1; graph._fit(); };
  $("#toggleDetailBtn").onclick = () => {
    if (state.incident) openEvidence(state.incident);
  };
  $("#memSearch").onclick = doRecall;
  $("#memQuery").onkeydown = (e) => { if (e.key === "Enter") doRecall(); };
  $("#memResetBtn").onclick = resetMemory;
  $("#runCampaign").onclick = runCampaign;
  $("#modalClose").onclick = closeModal;
  $("#modal").onclick = (e) => { if (e.target.id === "modal") closeModal(); };
  $("#paletteBtn").onclick = togglePalette;
  $("#paletteInput").oninput = renderPalette;
  $("#paletteInput").onkeydown = paletteKeys;
  $("#palette").onclick = (e) => { if (e.target.id === "palette") togglePalette(false); };
  $("#budgetSeg").onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.budget = Number(b.dataset.budget);
    $$("#budgetSeg button").forEach((x) => x.classList.toggle("on", x === b));
    toast(`disruption budget ${(state.budget * 100).toFixed(0)}% · re-optimising`);
    if (state.scenario) runScenario(state.scenario);
  };
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault(); togglePalette(); return;
    }
    if (e.key === "Escape") { closeModal(); togglePalette(false); return; }
    if (e.key === " ") { e.preventDefault(); replayTimeline(); return; }
    const map = { "1": "outbreak", "2": "plan", "3": "memory", "4": "model", "5": "eval" };
    if (map[e.key]) setTab(map[e.key]);
    if (e.key.toLowerCase() === "m") $("#memoryToggle").click();
    if (e.key.toLowerCase() === "e") executeFlow();
    if (e.key.toLowerCase() === "r" && state.scenario) runScenario(state.scenario);
  });
  window.addEventListener("resize", () => {
    if (state.incident) {
      renderIncident(state.incident, graph);
    }
  });
}

function setTab(tab) {
  state.tab = tab;
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
  if (tab === "memory") { refreshMemoryStats(); doRecall(); }
  if (tab === "model" || tab === "eval") refreshEval();
  if (tab === "plan" && state.incident) {
    requestAnimationFrame(() => {
      const ev = new Event("resize");
      window.dispatchEvent(ev);
    });
  }
}

function updateScrub(inc, t) {
  const horizon = inc.sim.horizon;
  let reached = 0;
  inc.sim.nodeStates.forEach((s) => {
    if (s.id !== inc.patientZero && s.t > 0 && s.t <= t && s.p > 0.004) reached += 1;
  });
  $("#reachedCount").textContent = reached;
  $("#clockReadout").textContent =
    `T+${String(Math.floor(t)).padStart(2, "0")}:${String(Math.floor((t % 1) * 60)).padStart(2, "0")}`;
  $("#scrub").style.background =
    `linear-gradient(90deg, #ff3b2f ${Math.round((t / horizon) * 100)}%, #2a2f2e ${Math.round((t / horizon) * 100)}%)`;
}

function replayTimeline() {
  if (!state.incident) return;
  $("#scrub").value = "0";
  graph.setClock(0);
  animateTimeline(state.incident, graph, 3600);
}

/* --------------------------------------------------------------- execution */
function executeFlow() {
  if (!state.incident) return;
  if (state.incident.status !== "open") {
    toast(`${state.incident.id} already ${state.incident.status}`);
    return;
  }
  openModal("ANALYST VERDICT", `
    <p style="color:#96988f;font-size:12px;line-height:1.7;max-width:620px">
      Execution is simulated: issue replacement, canary, verify, revoke, roll back if a
      health check fails. Your verdict is written to Hindsight as procedural memory.
    </p>
    <div style="display:flex;gap:8px;margin-top:18px">
      <button class="btn btn-primary" id="verdictOk">PORTFOLIO COMPLETE</button>
      <button class="btn btn-ghost" id="verdictBad">INCOMPLETE · ESCALATE</button>
    </div>
    <p class="evidence-note">Destructive actions require human approval. In production the
    adapter refuses to execute without an approver id and an idempotency key.</p>`);
  $("#verdictOk").onclick = () => runExecution("contained");
  $("#verdictBad").onclick = () => runExecution("incomplete");
}

async function runExecution(verdict) {
  closeModal();
  const stop = showLoading([
    "resolving every consumer of the target credential",
    "issuing replacements with narrower scope",
    "canary wave · health check · revoke old key",
    "writing episode, facts and directives to Hindsight",
  ]);
  try {
    const res = await api("/api/execute", {
      incidentId: state.incident.id,
      actor: "console-analyst",
      verdict,
    });
    stop();
    state.incident = res.incident;
    renderChoreo(res.execution);
    renderIncident(res.incident, graph);
    log("ok", `${res.incident.id} executed · ${res.execution.steps.length} steps · health ${(res.execution.health.authSuccess * 100).toFixed(2)}%`);
    log("mem", `retained episode · ${res.learned.facts} facts · ${res.learned.directives} directives · ${res.learned.edgesUpdated} edge posteriors updated`);
    toast(`Execution simulated · memory updated (${res.memory.records} records)`);
    setTab("plan");
    await refreshAfterRun();
  } catch (err) {
    stop();
    log("err", `execution failed: ${err.message}`);
    toast(err.message, true);
  }
}

/* ------------------------------------------------------------------ memory */
async function doRecall() {
  const q = $("#memQuery").value.trim() || "cut vertex";
  try {
    const r = await api(`/api/memory?q=${encodeURIComponent(q)}&limit=8`);
    renderRecall(r);
    log("recall", `“${q}” → ${r.results.length} memories in ${r.latencyMs}ms · ${r.backend || ""}`);
  } catch (err) { toast(`recall failed: ${err.message}`, true); }
}

function resetMemory() {
  openModal("RESET MEMORY", `
    <p style="color:#96988f;font-size:12px;line-height:1.7">
      This deletes every episode, fact, directive and edge posterior in the bank.
      The agent returns to a cold start. The graph and models are untouched.
    </p>
    <div style="display:flex;gap:8px;margin-top:18px">
      <button class="btn btn-danger-ghost" id="resetYes">DELETE THE BANK</button>
      <button class="btn btn-ghost" id="resetNo">CANCEL</button>
    </div>`);
  $("#resetNo").onclick = closeModal;
  $("#resetYes").onclick = async () => {
    try {
      const r = await api("/api/memory/reset", {});
      closeModal();
      await refreshMemoryStats();
      await refreshEval();
      log("warn", "memory bank reset · agent is cold again");
      toast("Memory bank deleted");
    } catch (err) { toast(err.message, true); }
  };
}

/* -------------------------------------------------------------------- eval */
async function runCampaign() {
  const stop = showLoading([
    "replaying six incidents in sequence",
    "cold plan vs warm plan on identical inputs",
    "retaining each episode and its directives",
    "measuring coverage@3 across the curve",
  ]);
  try {
    const res = await api("/api/campaign", { episodes: 6, trials: 900, budget: state.budget });
    state.eval = await api("/api/eval");
    renderModel(state.eval);
    renderEvalData(state.eval);
    stop();
    setTab("eval");
    log("ok", `replay complete · ${res.episodes} episodes · lift ${(res.gain * 100).toFixed(0)}pp`);
    toast(`Memory lift ${(res.gain * 100).toFixed(0)}pp · warm ${(res.warmMean * 100).toFixed(0)}% vs cold ${(res.coldMean * 100).toFixed(0)}%`);
    await refreshAfterRun();
  } catch (err) {
    stop();
    log("err", `campaign failed: ${err.message}`);
    toast(err.message, true);
  }
}

/* ----------------------------------------------------------------- evidence */
function openEvidence(inc) {
  const sim = inc.sim;
  const rows = sim.paths.slice(0, 8).map((p, i) => `
    <tr><td>${i + 1}</td><td>${Math.round(p.prob * 100)}%</td>
    <td>${fmt.usd0(p.conditionalLoss)}</td><td>${Math.round(p.minutes)}m</td>
    <td>${esc(p.labels.join(" → "))}</td></tr>`).join("");
  const edges = sim.edgeFire.filter((e) => e.fire > 0)
    .sort((a, b) => b.fire - a.fire).slice(0, 12);
  openModal("EVIDENCE TRACE", `
    <div class="micro">DOMINANT CHAINS · ${sim.trials.toLocaleString()} OUTBREAKS</div>
    <table class="evidence-table"><thead><tr><th>#</th><th>prob</th><th>loss</th>
      <th>time</th><th>chain</th></tr></thead><tbody>${rows || '<tr><td colspan="5">none</td></tr>'}</tbody></table>
    <div class="micro" style="margin-top:20px">EDGE TRANSMISSION POSTERIORS</div>
    <table class="evidence-table"><thead><tr><th>edge</th><th>fired</th></tr></thead><tbody>
      ${edges.map((e) => `<tr><td>${esc(e.id)}</td><td>${Math.round(e.fire * 100)}%</td></tr>`).join("")}
    </tbody></table>
    <p class="evidence-note">
      Simulation is a forecast with stated priors, not clairvoyance. The topology is
      synthetic. The algorithm, the posteriors and the optimisation are real.
      ${sim.beliefs.notes.join(" · ")}
    </p>`);
}

function onNodeSelect(node) {
  const inc = state.incident;
  const st = inc?.sim.nodeStates.find((s) => s.id === node.id);
  const incoming = (window.__graph?.edges || []).filter((e) => e.dst === node.id);
  const outgoing = (window.__graph?.edges || []).filter((e) => e.src === node.id);
  openModal(`NODE · ${node.label.toUpperCase()}`, `
    <div class="plan-summary" style="border-top:0;padding-top:0">
      <div><span class="micro">KIND</span><b style="font-size:12px">${node.kind.toUpperCase()}</b></div>
      <div><span class="micro">SERVICE</span><b style="font-size:12px">${esc(node.service || "—")}</b></div>
      <div><span class="micro">CRITICAL</span><b style="font-size:12px">${node.critical ? "YES" : "NO"}</b></div>
      <div><span class="micro">REACH P</span><b style="font-size:12px">${st ? Math.round(st.p * 100) + "%" : "—"}</b></div>
      <div><span class="micro">VALUE</span><b style="font-size:12px">${node.value ? fmt.usd0(node.value) : "—"}</b></div>
    </div>
    <p style="color:#96988f;font-size:11px;line-height:1.7">${esc(node.detail || "")}</p>
    <div class="micro" style="margin-top:14px">INBOUND (${incoming.length})</div>
    <table class="evidence-table"><tbody>
      ${incoming.slice(0, 10).map((e) => `<tr><td>${esc(e.rel)}</td><td>${(e.p * 100).toFixed(0)}%</td><td>${(e.conf * 100).toFixed(0)}% conf</td><td>${esc(e.source)}</td></tr>`).join("")}
    </tbody></table>
    <div class="micro" style="margin-top:14px">OUTBOUND (${outgoing.length})</div>
    <table class="evidence-table"><tbody>
      ${outgoing.slice(0, 10).map((e) => `<tr><td>${esc(e.rel)}</td><td>${(e.p * 100).toFixed(0)}%</td><td>${(e.conf * 100).toFixed(0)}% conf</td><td>${esc(e.source)}</td></tr>`).join("")}
    </tbody></table>`);
}

/* ----------------------------------------------------------------- palette */
const COMMANDS = [
  { label: "Run the selected scenario", hint: "R", run: () => runScenario(state.scenario) },
  { label: "Execute the containment plan", hint: "E", run: executeFlow },
  { label: "Toggle Hindsight memory", hint: "M", run: () => $("#memoryToggle").click() },
  { label: "Open the memory bank", hint: "3", run: () => setTab("memory") },
  { label: "Open the model card", hint: "4", run: () => setTab("model") },
  { label: "Replay six incidents (learning curve)", hint: "EVAL", run: runCampaign },
  { label: "Show evidence trace", hint: "EVIDENCE", run: () => state.incident && openEvidence(state.incident) },
  { label: "Reset memory", hint: "DANGER", run: resetMemory },
];

function togglePalette(force) {
  const el = $("#palette");
  const on = force === undefined ? !el.classList.contains("on") : force;
  el.classList.toggle("on", on);
  if (on) { $("#paletteInput").value = ""; renderPalette(); setTimeout(() => $("#paletteInput").focus(), 30); }
}

function renderPalette() {
  const q = $("#paletteInput").value.toLowerCase();
  const list = $("#paletteList");
  const cmds = COMMANDS.filter((c) => c.label.toLowerCase().includes(q));
  list.innerHTML = cmds.map((c, i) =>
    `<div class="palette-item ${i === 0 ? "on" : ""}" data-i="${COMMANDS.indexOf(c)}">
      <span>${esc(c.label)}</span><span class="pi-hint">${esc(c.hint)}</span></div>`).join("");
  list.querySelectorAll(".palette-item").forEach((n) => {
    n.onclick = () => { COMMANDS[Number(n.dataset.i)].run(); togglePalette(false); };
  });
}

function paletteKeys(e) {
  const items = [...$$("#paletteList .palette-item")];
  let idx = items.findIndex((n) => n.classList.contains("on"));
  if (e.key === "ArrowDown") {
    e.preventDefault();
    items.forEach((n) => n.classList.remove("on"));
    idx = (idx + 1) % Math.max(1, items.length);
    items[idx]?.classList.add("on");
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    items.forEach((n) => n.classList.remove("on"));
    idx = (idx - 1 + items.length) % Math.max(1, items.length);
    items[idx]?.classList.add("on");
  } else if (e.key === "Enter") {
    e.preventDefault();
    items[idx]?.click();
  }
}

/* ------------------------------------------------------------------- start */
init().catch((err) => {
  console.error(err);
  $("#bootLog").innerHTML += `<div style="color:#ff3b2f">boot failed: ${esc(err.message)}</div>`;
  $("#bootEnter").disabled = false;
});
