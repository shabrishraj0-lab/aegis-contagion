/* Rendering functions. Every panel is a pure function of state → DOM. */
import { $, esc, fmt, el, relLabel, relColor } from "./api.js";
import {
  drawSpark, drawFrontier, drawLossCurve, drawCalibration,
  drawLearning, drawEval,
} from "./charts.js";

const KIND_LABEL = {
  credential: "CREDENTIAL", workload: "WORKLOAD", human: "HUMAN",
  repo: "REPOSITORY", pipeline: "PIPELINE", role: "CLOUD ROLE",
  datastore: "DATA STORE", ai_endpoint: "AI ENDPOINT", idp: "IDENTITY PROVIDER",
};

export function renderScenarios(scenarios, onPick) {
  const list = $("#scenarioList");
  list.innerHTML = "";
  scenarios.forEach((s, i) => {
    const b = el("button", "scenario");
    b.dataset.scenario = s.id;
    b.innerHTML = `<span class="si">${String(i + 1).padStart(2, "0")}</span>
      <span class="sl">${esc(s.label)}</span>
      <span class="ss">${esc(s.signal)}</span>
      <span class="sm"><span class="tag">${esc(s.estate.toUpperCase())}</span>
      <span class="tag red">${esc(s.p0.replace("key-", "").slice(0, 14))}</span></span>`;
    b.onclick = () => onPick(s);
    list.appendChild(b);
  });
  $("#scenarioCount").textContent = String(scenarios.length).padStart(2, "0");
}

export function markScenario(id) {
  [...document.querySelectorAll(".scenario")].forEach((b, i) => {
    b.classList.toggle("on", b.dataset.scenario === id);
  });
}

export function renderIncidents(incidents, onOpen) {
  const list = $("#incidentList");
  const count = incidents?.length || 0;
  $("#incidentCount").textContent = String(count).padStart(2, "0");
  if (!count) {
    list.innerHTML = `<div class="empty-line">no outbreak run this session</div>`;
    return;
  }
  list.innerHTML = "";
  incidents.forEach((inc) => {
    const row = el("div", "incident");
    row.innerHTML = `<span class="iid">${esc(inc.id)}</span>
      <span class="inm">${esc(inc.patientZeroLabel)}</span>
      <span class="ist ${esc(inc.status)}"></span>`;
    row.onclick = () => onOpen(inc);
    list.appendChild(row);
  });
}

export function renderIncident(inc, graph) {
  if (!inc) return;
  const sim = inc.sim;
  const k = sim.kpis;
  $("#kpiR0").textContent = k.r0.toFixed(2);
  $("#kpiTtc").textContent = k.ttcP50 == null ? "—" : `${Math.round(k.ttcP50)}m`;
  $("#kpiReach").textContent = `${Math.round(k.reachProbability * 100)}%`;
  $("#kpiLoss").textContent = fmt.usd0(k.lossP50);
  $("#lossRange").textContent = `p90 ${fmt.usd0(k.lossP90)} · p99 ${fmt.usd0(k.lossP99)}`;
  $("#kpiSpread").textContent = k.credentialSpread.toFixed(1);
  $("#beliefBadge").textContent = inc.memoryUsed ? "LEARNED" : "COLD";
  $("#beliefBadge").className = "badge " + (inc.memoryUsed ? "learned" : "");
  $("#pathCount").textContent = String(sim.paths.length).padStart(2, "0");
  $("#portfolioMeta").textContent =
    `${inc.plan.actions.length} CUTS · ${Math.round(inc.plan.coverage * 100)}%`;
  $("#graphTitle").textContent =
    `${inc.estateName.toUpperCase()} · ${sim.trials.toLocaleString()} OUTBREAKS`;
  $("#ticker").textContent =
    `${inc.id} · ${inc.patientZeroLabel.toUpperCase()} · R₀ ${k.r0} · ` +
    `${Math.round(inc.plan.coverage * 100)}% COVERAGE · ${inc.memoryUsed ? "HINDSIGHT WARM" : "COLD START"}`;
  $("#dockBackend").textContent = (inc.memoryMode || "local").toUpperCase();
  $("#dockLatency").textContent = `${inc.latencyMs} ms`;
  $("#dockRecall").textContent = `${inc.recalled.length} recalled`;
  $("#memBackend").textContent = (inc.memoryMode || "local").toUpperCase();
  $("#memDelta").textContent = inc.memoryUsed ? "priors applied" : "not consulted";
  $("#memCount").textContent = String(inc.recalled.length);

  renderSpark(sim.curve);
  renderPaths(sim.paths, inc);
  renderPortfolio(inc);
  renderPlan(inc);
  renderTrace(inc.trace);
  renderDirectives(inc);
  renderMemo(inc);
  renderBeliefDelta(inc);
  if (graph) {
    graph.setIncident(inc);
    graph.setRecalled(recalledNodeIds(inc));
  }
  const stillOpen = inc.status === "open";
  $("#executeBtn").disabled = !stillOpen;
  $("#planExecuteBtn").disabled = !stillOpen;
  $("#executeBtn").textContent = stillOpen ? "EXECUTE CONTAINMENT" : "AUDIT SEALED";
  $("#planExecuteBtn").textContent = stillOpen ? "EXECUTE PLAN" : "AUDIT SEALED";
  $("#scrub").max = String(sim.horizon);
  $("#scrub").value = "0";
  updateTimeline(inc, 0);
  animateTimeline(inc, graph);
}

function recalledNodeIds(inc) {
  const ids = new Set();
  const known = new Set((window.__graph?.nodes || []).map((n) => n.id));
  for (const r of inc.recalled || []) {
    for (const e of r.entities || []) {
      if (known.has(e)) ids.add(e);
    }
  }
  return [...ids];
}

export function renderSpark(curve) {
  drawSpark($("#curveSpark"), curve, "#ff3b2f");
}

export function renderPaths(paths, inc) {
  const list = $("#pathList");
  if (!paths?.length) {
    list.innerHTML = `<div class="empty-line">no completed chain inside the horizon</div>`;
    return;
  }
  list.innerHTML = "";
  paths.slice(0, 6).forEach((p) => {
    const row = el("div", "path");
    const chain = p.labels.map((l) => `<em>${esc(l)}</em>`).join(
      ` <span class="rel">→</span> `);
    row.innerHTML = `<div class="path-top">
        <span class="path-prob">${Math.round(p.prob * 100)}%</span>
        <span class="path-min">${Math.round(p.minutes)}m</span>
        <span class="path-loss">${fmt.usd0(p.conditionalLoss)}</span>
      </div><div class="path-chain">${chain}</div>`;
    row.onclick = () => openPathModal(p, inc);
    list.appendChild(row);
  });
}

export function renderPortfolio(inc) {
  const list = $("#planList");
  list.innerHTML = "";
  inc.plan.actions.forEach((a, i) => {
    const card = el("div", "action-card");
    card.innerHTML = `
      <div class="ac-head"><span class="ac-idx">${i + 1}</span>
      <span class="ac-label">${esc(a.label)}</span><span class="ac-icon">${a.icon || "↻"}</span></div>
      <div class="ac-target">${esc(a.node_label)} · ${a.minutes}m · ${a.reversible ? "reversible" : "irreversible"}</div>
      <div class="ac-bar"><i style="width:${Math.round(a.coverage * 100)}%"></i></div>
      <div class="ac-meta"><span>cuts <b>${Math.round(a.coverage * 100)}%</b></span>
      <span>disruption <b>${(a.disruption * 100).toFixed(1)}%</b></span>
      <span>cost <b>${fmt.usd0(a.cost)}</b></span></div>`;
    card.onclick = () => openActionModal(a, inc);
    list.appendChild(card);
  });
  const total = el("div", "portfolio-total");
  total.innerHTML = `<div><b>${Math.round(inc.plan.coverage * 100)}%</b>
      <span> of residual paths cut</span></div>
      <span>${(inc.plan.disruption * 100).toFixed(1)}% disruption · ${fmt.usd0(inc.plan.cost)}</span>`;
  list.appendChild(total);
}

function renderPlan(inc) {
  const plan = inc.plan;
  const combos = plan.search.combinationsEvaluated;
  $("#planFormula").textContent =
    `greedy + exhaustive shortlist · ${combos.toLocaleString()} portfolio${combos === 1 ? "" : "s"}`;
  const box = $("#planDetail");
  if (!plan.actions.length) {
    box.innerHTML = `<div class="empty-line">${plan.pathsTotal
      ? "no intervention fits the budget"
      : "no residual chain reaches a critical asset inside the horizon"}</div>`;
  } else {
    box.innerHTML = "";
    plan.actions.forEach((a, i) => {
      const row = el("div", "plan-row");
      row.innerHTML = `<div class="pr-idx">${String(i + 1).padStart(2, "0")}</div>
        <div class="pr-kind">${esc(a.label)}<span>${esc(a.node_label)} · ${a.minutes} min</span></div>
        <div class="pr-rationale">${esc(a.rationale)}</div>
        <div><div class="pr-bar"><i style="width:${Math.round(a.coverage * 100)}%"></i></div></div>
        <div class="pr-pct">${Math.round(a.coverage * 100)}% <span style="color:#5e625d">/ ${(a.disruption * 100).toFixed(1)}%</span></div>
        <div class="pr-cost">${fmt.usd0(a.cost)}<br>${a.reversible ? "reversible" : "no rollback"}</div>`;
      row.onclick = () => openActionModal(a, inc);
      box.appendChild(row);
    });
    const summary = el("div", "plan-summary");
    summary.innerHTML = `<div><span class="micro">COVERAGE</span><b class="good">${Math.round(plan.coverage * 100)}%</b></div>
      <div><span class="micro">DISRUPTION</span><b>${(plan.disruption * 100).toFixed(1)}%</b></div>
      <div><span class="micro">COST</span><b>${fmt.usd0(plan.cost)}</b></div>
      <div><span class="micro">PATHS CUT</span><b>${plan.pathsBlocked}/${plan.pathsTotal}</b></div>
      <div><span class="micro">NAIVE ROTATION</span><b style="color:#ffb020">${Math.round(plan.baseline.coverage * 100)}%</b></div>`;
    box.appendChild(summary);
  }
  requestAnimationFrame(() => {
    drawFrontier($("#frontierChart"), plan.budgetCurve, plan.disruption);
    renderBaselineBars(inc);
    $("#frontierFoot").textContent =
      `chosen portfolio sits at ${(plan.disruption * 100).toFixed(1)}% disruption; ` +
      `the knee is where coverage stops paying for itself.`;
  });
  renderChoreo(inc.execution);
}

function renderBaselineBars(inc) {
  const box = $("#baselineBars");
  const plan = inc.plan;
  const cold = inc.cold?.coverage ?? null;
  const warm = plan.truthCoverage ?? plan.coverage;
  const naive = plan.baselineTruth?.coverage ?? plan.baseline.coverage;
  const rows = [
    ["NAIVE ROTATION", naive, "naive",
      "rotate the three highest-degree credentials"],
    ["AGENT · COLD START", cold, "cold", "no memory forecast · scored held-out"],
    ["AGENT · HINDSIGHT", warm, "warm",
      `${inc.recalled.length} records recalled · held-out score`],
  ];
  box.innerHTML = "";
  rows.forEach(([label, value, cls, foot]) => {
    const row = el("div", "bar-row");
    row.innerHTML = `<div class="br-top"><span>${label}</span>
      <b style="color:${cls === "warm" ? "#4ce3d2" : cls === "naive" ? "#ffb020" : "#8e918a"}">
      ${value == null ? "—" : Math.round(value * 100) + "%"}</b></div>
      <div class="bar-track"><i class="${cls}" style="width:${value == null ? 0 : Math.round(value * 100)}%"></i></div>
      <div class="br-foot">${esc(foot)}</div>`;
    box.appendChild(row);
  });
}

export function renderTrace(trace) {
  const box = $("#traceList");
  if (!trace?.length) { box.innerHTML = `<div class="empty-line">—</div>`; return; }
  box.innerHTML = "";
  trace.forEach((t) => {
    const row = el("div", "trace-row");
    row.innerHTML = `<span class="tr-step">${esc(t.step)}</span>
      <span class="tr-detail">${esc(t.detail)}</span>
      <span class="tr-ms">${t.ms}ms</span>`;
    box.appendChild(row);
  });
}

export function renderDirectives(inc) {
  const box = $("#directiveList");
  const dirs = inc.beliefs?.directives || [];
  const applied = inc.plan.directivesApplied || [];
  if (!dirs.length && !applied.length) {
    box.innerHTML = `<div class="empty-line">none — cold plan has no analyst constraints</div>`;
    $("#directivesFull").innerHTML = `<div class="empty-line">none yet</div>`;
    return;
  }
  box.innerHTML = "";
  dirs.forEach((d) => {
    const n = el("div", "directive" + (d.kind === "avoid_rotation" ? " avoid" : ""));
    n.innerHTML = `${esc(d.text)}<span>${esc(d.kind.toUpperCase())} · ${esc(d.incident || "")}</span>`;
    box.appendChild(n);
  });
  const full = $("#directivesFull");
  full.innerHTML = "";
  dirs.forEach((d) => {
    const n = el("div", "directive" + (d.kind === "avoid_rotation" ? " avoid" : ""));
    n.innerHTML = `${esc(d.text)}<span>${esc((d.kind || "").toUpperCase())} · ${esc(d.incident || "")}</span>`;
    full.appendChild(n);
  });
}

export function renderChoreo(execution) {
  const box = $("#choreo");
  if (!execution) {
    box.innerHTML = `<div class="empty-line">execute the plan to simulate the runbook</div>`;
    return;
  }
  box.innerHTML = "";
  execution.steps.forEach((s, i) => {
    const row = el("div", "choreo-step");
    row.style.animationDelay = `${i * 22}ms`;
    row.classList.add("reveal");
    row.innerHTML = `<div class="cs-dot"></div>
      <div class="cs-phase">${esc(s.phase.toUpperCase())}</div>
      <div class="cs-action">${esc(s.action)} · ${esc(s.target)}
        <span>${esc(s.detail)}</span></div>
      <div class="cs-min">${s.minutes}m</div>`;
    box.appendChild(row);
  });
  const foot = el("div", "card-foot");
  foot.textContent = `${execution.steps.length} steps · ${execution.simulatedMinutes} simulated minutes · ` +
    `auth success ${(execution.health.authSuccess * 100).toFixed(2)}% · ` +
    `${execution.audit.length} audit entries`;
  box.appendChild(foot);
}

export function renderMemo(inc) {
  const memo = inc.memo;
  $("#lmDraft").textContent = memo.draft || "—";
  const v = memo.validator;
  const f = v.final || { checked: 0, accepted: 0, rejected: [] };
  $("#validatorVerdict").innerHTML =
    `draft ${v.accepted}/${v.checked} grounded · final ${f.accepted}/${f.checked} verified`;
  $("#validatorVerdict").className = "micro " + (f.rejected.length ? "warn" : "ok");
  const list = $("#validatorList");
  list.innerHTML = "";
  (v.checks || []).slice(0, 6).forEach((c) => {
    const chip = el("span", "vchip " + (c.ok ? "ok" : "bad"), esc(c.claim));
    list.appendChild(chip);
  });
  if (!v.checked) list.innerHTML = `<div class="empty-line">draft had no numeric claims</div>`;
  if (f.checked) {
    const foot = el("div", "validator-foot",
      `deterministic memo re-checked against simulation · ` +
      `${f.checked} claims · ${f.accepted} present`);
    list.appendChild(foot);
  }
  $("#lmStats").innerHTML = `
    <div><span class="micro">PARAMS</span><b>${fmt.int(memo.lm.params)}</b></div>
    <div><span class="micro">PERPLEXITY</span><b>${memo.lm.perplexity}</b></div>
    <div><span class="micro">TRAINED</span><b>LOCALLY</b></div>
    <div><span class="micro">MEMORY</span><b>${inc.memoryUsed ? "WARM" : "COLD"}</b></div>`;
}

export function renderBeliefDelta(inc) {
  const box = $("#memDiff");
  const b = inc.beliefs || {};
  if (!inc.memoryUsed) {
    box.innerHTML = `<div class="empty-line">memory disabled: no priors, no directives, no learning</div>`;
    return;
  }
  const rows = [
    ["RECALLED", `${inc.recalled.length} records`, `${inc.memo.memoryLine}`],
    ["EDGE PRIORS", `${b.edges} overrides`, "Beta posteriors from observed transmissions"],
    ["RELATION PRIORS", `${b.relations} groups`, "per-relation transmission correction"],
    ["DIRECTIVES", `${(b.directives || []).length} active`, "constraints applied to the portfolio"],
    ["ACTION PRIORS", `${Object.keys(b.actionPriors || {}).length} kinds`, "coverage per unit disruption"],
    ["GNN", "blended", "edge calibration ×0.45 exponent"],
  ];
  box.innerHTML = "";
  rows.forEach(([a, c, d]) => {
    const r = el("div", "diff-row");
    r.innerHTML = `<b>${a}</b><span><strong style="color:#b9bdb5;font-weight:400">${esc(c)}</strong> · ${esc(d)}</span>`;
    box.appendChild(r);
  });
}

function updateTimeline(inc, t) {
  const nodeStates = inc?.sim?.nodeStates || [];
  let reached = 0;
  const horizon = inc?.sim?.horizon || 90;
  nodeStates.forEach((s) => {
    if (s.id === inc.patientZero) return;
    if (s.t > 0 && s.t <= t && s.p > 0.004) reached += 1;
  });
  $("#reachedCount").textContent = reached;
  $("#clockReadout").textContent = `T+${String(Math.floor(t)).padStart(2, "0")}:${String(Math.floor((t % 1) * 60)).padStart(2, "0")}`;
  const phase = t <= 0 ? "PATIENT ZERO" : t < 15 ? "INITIAL ACCESS" :
    t < 35 ? "LATERAL MOVEMENT" : t < 60 ? "PRIVILEGE ESCALATION" : "CROWN JEWEL REACHED";
  $("#phaseLabel").textContent = inc ? phase : "AWAITING PATIENT ZERO";
  const pct = Math.round((t / horizon) * 100);
  $("#scrub").style.background =
    `linear-gradient(90deg, #ff3b2f ${pct}%, #2a2f2e ${pct}%)`;
}

export function animateTimeline(inc, graph, duration = 3200) {
  const horizon = inc.sim.horizon;
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    const t = horizon * eased;
    $("#scrub").value = String(t);
    updateTimeline(inc, t);
    graph?.setClock(t);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

export function renderMemoryStats(stats, records) {
  const s = stats?.stats || stats || {};
  $("#memBackend").textContent = (s.mode || "local").toUpperCase();
  $("#memCount").textContent = String(s.records ?? 0);
  $("#memGraphMeta").textContent =
    `${s.byType?.episode || 0} episodes · ${s.byType?.fact || 0} facts · ${s.byType?.directive || 0} directives`;
  $("#dockBackend").textContent = (s.mode || "local").toUpperCase();
  $("#memDelta").textContent =
    `${s.edgesLearned || 0} edge beliefs · ${s.directives || 0} directives`;
}

export function renderRecall(data) {
  const box = $("#memResults");
  const results = data?.results || [];
  if (!results.length) {
    box.innerHTML = `<div class="empty-line">nothing recalled — the bank is empty or the query has no overlap</div>`;
    return;
  }
  box.innerHTML = "";
  results.forEach((r) => {
    const card = el("div", "mem-card");
    const ch = r.channels || {};
    card.innerHTML = `<div class="mc-top"><span class="mc-type">${esc(r.type.toUpperCase())}</span>
      <span class="mc-score">${(r.score * 100).toFixed(0)}%</span></div>
      <div class="mc-text">${esc(r.text)}</div>
      <div class="mc-ch">
        <span class="ch">dense ${(ch.dense ?? 0).toFixed(2)}</span>
        <span class="ch">kw ${(ch.keyword ?? 0).toFixed(1)}</span>
        <span class="ch">graph ${(ch.graph ?? 0).toFixed(1)}</span>
        <span class="ch">time ${(ch.temporal ?? 0).toFixed(2)}</span>
      </div>`;
    box.appendChild(card);
  });
  $("#dockRecall").textContent = `${results.length} recalled · ${data.latencyMs}ms`;
}

export function renderModel(evalData) {
  const g = evalData.gnn || {};
  const node = g.node || {}, edge = g.edge || {};
  $("#gnnAp").textContent = node.ap ?? "—";
  $("#gnnEdgeAp").textContent = edge.ap ?? "—";
  $("#gnnEce").textContent = node.ece ?? "—";
  $("#gnnBrier").textContent = node.brier ?? "—";
  $("#gnnNote").textContent =
    `${fmt.int(g.params || 0)} parameters · ${g.epochs || "—"} epochs · ` +
    `${g.trainSeconds || "—"}s training · temperature ${g.temperature || "—"} · ` +
    `${g.split || ""}. Labels: ${g.labels || ""}.`;
  $("#archFoot").textContent =
    `node AP ${node.ap} vs base rate ${node.positiveRate} · edge AP ${edge.ap} · ` +
    `ECE ${node.ece} after temperature scaling · holdout ${g.holdoutNodes} nodes / ${g.holdoutEdges} edges`;
  $("#archDiagram").innerHTML = [
    ["NODE FEATURES", "21 dims · kind, privilege, exposure, centrality"],
    ["RELATION AGG", "6 groups · secret, identity, assume, data, deploy, invoke"],
    ["LAYER 1", "48 units · ReLU · weighted mean"],
    ["LAYER 2", "48 units · residual context"],
    ["NODE HEAD", "P(reached within horizon)"],
    ["EDGE HEAD", "P(edge transmits)"],
  ].map(([a, b], i) => `
    <div class="arch-node ${i === 4 || i === 5 ? "hot" : ""}">
      <b>${a}</b><span>${b}</span></div>
    ${i < 5 ? '<div class="arch-arrow">→</div>' : ""}`).join("");
  requestAnimationFrame(() => {
    drawLossCurve($("#lossChart"), g.lossCurve || []);
    drawCalibration($("#calChart"), node.calibration || []);
    drawLearning($("#learningChart"),
      (evalData.campaign?.series || []).filter((s) => (s.pathsTotal ?? 1) > 0));
  });
  $("#learningFoot").textContent = evalData.campaign
    ? `${evalData.campaign.episodes} episodes · warm ${(evalData.campaign.warmMean * 100).toFixed(0)}% vs cold ${(evalData.campaign.coldMean * 100).toFixed(0)}%`
    : "run the replay to populate";
  renderEvalData(evalData);
  const chips = $("#modelChips");
  chips.innerHTML = `
    <span class="chip on">NUMPY ONLY</span>
    <span class="chip">NO PRETRAINED WEIGHTS</span>
    <span class="chip">NO INFERENCE API</span>`;
  if (evalData.lm) {
    $("#lmStats").innerHTML = `
      <div><span class="micro">PARAMS</span><b>${fmt.int(evalData.lm.params)}</b></div>
      <div><span class="micro">PERPLEXITY</span><b>${evalData.lm.perplexity}</b></div>
      <div><span class="micro">DOCS</span><b>${fmt.int(evalData.lm.docs)}</b></div>
      <div><span class="micro">STEPS</span><b>${evalData.lm.steps}</b></div>
      <div><span class="micro">TRAIN</span><b>${evalData.lm.trainSeconds}s</b></div>`;
  }
}

export function renderEvalData(evalData) {
  const camp = evalData.campaign;
  const box = $("#evalTable");
  const bars = $("#evalBars");
  if (!camp) {
    $("#evalBig").textContent = "—";
    box.innerHTML = `<div class="empty-line">no replay yet</div>`;
    bars.innerHTML = `<div class="empty-line">run the replay</div>`;
    $("#evalMeta").textContent = "—";
    return;
  }
  const lift = (camp.warmMean - camp.coldMean) * 100;
  $("#evalBig").innerHTML = `${lift > 0 ? "+" : ""}${lift.toFixed(1)}<small>percentage points of held-out coverage on memory-eligible episodes</small>`;
  const memoryCount = camp.memoryEpisodes ??
    camp.series.filter((s) => (s.pathsTotal ?? 1) > 0 && s.recalled > 0).length;
  $("#evalMeta").textContent =
    `${camp.episodes} episodes · ${memoryCount} with memory · warm ${(camp.warmMean * 100).toFixed(0)}% · cold ${(camp.coldMean * 100).toFixed(0)}% · naive ${(camp.naiveMean * 100).toFixed(0)}%`;
  bars.innerHTML = "";
  [["NAIVE", camp.naiveMean, "naive"], ["COLD", camp.coldMean, "cold"],
   ["HINDSIGHT", camp.warmMean, "warm"]].forEach(([label, v, cls]) => {
    const row = el("div", "bar-row");
    row.innerHTML = `<div class="br-top"><span>${label}</span><b>${Math.round(v * 100)}%</b></div>
      <div class="bar-track"><i class="${cls}" style="width:${Math.round(v * 100)}%"></i></div>`;
    bars.appendChild(row);
  });
  box.innerHTML = `<table><thead><tr><th>#</th><th>incident</th><th>recalled</th>
    <th>paths</th><th>cold</th><th>warm</th><th>Δ</th><th>actions</th></tr></thead><tbody></tbody></table>`;
  const tbody = box.querySelector("tbody");
  camp.series.forEach((s) => {
    const hasPath = (s.pathsTotal ?? 1) > 0;
    const d = Math.round((s.warmCoverage - s.coldCoverage) * 100);
    const tr = el("tr");
    tr.innerHTML = `<td>${String(s.index).padStart(2, "0")}</td>
      <td>${esc(s.scenario)}</td><td>${s.recalled}</td>
      <td>${s.pathsTotal ?? "—"}</td>
      <td>${hasPath ? Math.round(s.coldCoverage * 100) + "%" : "—"}</td>
      <td>${hasPath ? Math.round(s.warmCoverage * 100) + "%" : "—"}</td>
      <td class="${hasPath && d >= 0 ? "pos" : "neg"}">${hasPath ? (d >= 0 ? "+" : "") + d + "pp" : "no path"}</td>
      <td>${esc(hasPath ? s.actions.join(", ") : "no reachable critical asset")}</td>`;
    tbody.appendChild(tr);
  });
  $("#evalAblations").innerHTML = [
    ["Full agent (memory + GNN + optimiser)", `${(camp.warmMean * 100).toFixed(0)}%`],
    ["Memory ablated (cold start)", `${(camp.coldMean * 100).toFixed(0)}%`],
    ["Portfolio ablated (naive top-k rotation)", `${(camp.naiveMean * 100).toFixed(0)}%`],
    ["Disruption budget", "8%"],
    ["Simulator", "900 outbreaks / episode"],
  ].map(([a, b]) => `<div class="abl"><span>${a}</span><b>${b}</b></div>`).join("");
  requestAnimationFrame(() => drawEval($("#evalChart"),
    camp.series.filter((s) => (s.pathsTotal ?? 1) > 0)));
}

export function renderPlanDesign(design) {
  const box = $("#designList");
  box.innerHTML = "";
  (design?.layers || []).forEach((l, i) => {
    const row = el("div", "design-row");
    row.innerHTML = `<span class="dr-n">${String(i + 1).padStart(2, "0")}</span>
      <span class="dr-name">${esc(l.name)}</span>
      <span class="dr-detail">${esc(l.detail)}</span>`;
    box.appendChild(row);
  });
}

/* ------------------------------ modals ------------------------------ */
export function openModal(title, html) {
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = html;
  $("#modal").classList.add("on");
}

export function closeModal() { $("#modal").classList.remove("on"); }

function openPathModal(path, inc) {
  const edgeMap = new Map(inc.sim.edgeFire.map((e) => [e.id, e.fire]));
  const estate = window.__graph;
  const nodeMap = new Map((estate?.nodes || []).map((n) => [n.id, n]));
  const edges = path.edges.map((id, i) => {
    const [src, rel, dst] = id.split("|");
    const edge = estate?.edges.find((e) => e.id === id);
    return `<tr><td>${i + 1}</td><td>${esc(nodeMap.get(src)?.label || src)}</td>
      <td><span class="pill" style="color:${relColor[rel] || "#999"}">${esc(relLabel[rel] || rel)}</span></td>
      <td>${esc(nodeMap.get(dst)?.label || dst)}</td>
      <td>${edge ? (edge.p * 100).toFixed(0) + "%" : "—"}</td>
      <td>${edge ? (edge.conf * 100).toFixed(0) + "%" : "—"}</td>
      <td>${edge?.source === "inferred" ? "INFERRED" : "OBSERVED"}</td>
      <td>${Math.round((edgeMap.get(id) || 0) * 100)}%</td></tr>`;
  }).join("");
  openModal(`PATH · ${Math.round(path.prob * 100)}% of outbreaks`, `
    <div class="micro">CHAIN</div>
    <p style="font-family:'JetBrains Mono';font-size:10px;line-height:1.9;color:#b9bdb5">
      ${path.labels.map((l) => esc(l)).join(' <span style="color:#ff3b2f">→</span> ')}
    </p>
    <table class="evidence-table"><thead><tr><th>#</th><th>from</th><th>relation</th>
      <th>to</th><th>p</th><th>conf</th><th>source</th><th>fired</th></tr></thead>
      <tbody>${edges}</tbody></table>
    <p class="evidence-note">
      Probability of this chain completing: <b style="color:#ffb020">${(path.prob * 100).toFixed(1)}%</b> ·
      conditional loss <b>${fmt.usd(path.conditionalLoss)}</b> ·
      median completion <b>${Math.round(path.minutes)} minutes</b>.
      Every hop traces to an edge with a confidence and a last-verified timestamp.
    </p>`);
}

function openActionModal(a, inc) {
  const blocked = inc.plan.actions.filter((x) => x.target === a.target).length;
  openModal(`INTERVENTION · ${a.label.toUpperCase()}`, `
    <div class="plan-summary" style="border-top:0;padding-top:0">
      <div><span class="micro">TARGET</span><b style="font-size:13px">${esc(a.node_label)}</b></div>
      <div><span class="micro">COVERAGE</span><b class="good">${Math.round(a.coverage * 100)}%</b></div>
      <div><span class="micro">DISRUPTION</span><b>${(a.disruption * 100).toFixed(1)}%</b></div>
      <div><span class="micro">TIME</span><b>${a.minutes}m</b></div>
      <div><span class="micro">COST</span><b>${fmt.usd0(a.cost)}</b></div>
    </div>
    <p style="color:#96988f;font-size:11px;line-height:1.7">${esc(a.rationale)}</p>
    <p class="evidence-note">${a.reversible
      ? "Reversible: the old credential remains valid until the health window closes."
      : "Irreversible: only safe when the consumer list is complete."}
      ${blocked} intervention(s) target this node in the chosen portfolio.
      Expected loss blocked: <b>${fmt.usd(a.expectedBlocked)}</b>.</p>`);
}

/* ------------------------------ dock ------------------------------ */
let logLines = 0;
export function log(kind, text) {
  const box = $("#consoleLog");
  const line = el("div", "logline " + kind);
  const now = new Date();
  const ts = now.toLocaleTimeString("en-GB", { hour12: false });
  line.innerHTML = `<span class="lt">${ts}</span><span class="lk">${esc(kind.toUpperCase())}</span><span class="lx">${esc(text)}</span>`;
  box.appendChild(line);
  logLines += 1;
  if (logLines > 160) box.removeChild(box.firstChild);
  box.scrollTop = box.scrollHeight;
}

let toastTimer = null;
export function toast(text, err = false) {
  const t = $("#toast");
  t.textContent = text;
  t.className = "toast on" + (err ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = "toast"; }, 3200);
}

export function showLoading(steps) {
  const box = $("#loadingSteps");
  box.innerHTML = "";
  steps.forEach((s) => {
    const d = el("div", "loading-step", esc(s));
    box.appendChild(d);
  });
  $("#loading").classList.add("on");
  let i = 0;
  const timer = setInterval(() => {
    const items = [...box.children];
    if (i < items.length) items[i].classList.add("done");
    i += 1;
    if (i > items.length) clearInterval(timer);
  }, 260);
  return () => { clearInterval(timer); $("#loading").classList.remove("on"); };
}

export function updateClock() {
  const now = new Date();
  $("#clock").innerHTML = `${now.toLocaleTimeString("en-GB", { hour12: false })}<span>Z</span>`;
}
