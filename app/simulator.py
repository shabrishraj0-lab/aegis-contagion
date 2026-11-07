"""Vectorised Monte Carlo outbreak simulator.

A credential compromise is a diffusion process on the identity graph.  Each
edge has a per-attempt success probability and a time cost; each infected node
has a detection hazard (the defender's dwell time).  We simulate thousands of
independent outbreaks with NumPy:

  * per-edge time-to-success  ~ Geometric(p) * t
  * per-node detection window ~ Exponential(dwell_mean)
  * infection times via a vectorised Bellman-Ford relaxation, so cycles and
    multiple paths are handled without hand-waving.

Outputs are distributions, not point claims: R0, time-to-crown-jewel, P50/P90
loss, aggregated attack paths, next-hop marginals, edge fire rates and the
infection curve used by the UI scrubber.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .estate import Estate
from .impact import asset_loss

KIND_DWELL = {
    "credential": 34.0, "workload": 48.0, "human": 70.0, "repo": 55.0,
    "pipeline": 31.0, "role": 44.0, "datastore": 22.0, "ai_endpoint": 38.0,
    "idp": 26.0,
}


@dataclass
class Beliefs:
    """Everything memory injects into a run.  Cold start is the empty object."""
    edge_mult: dict[str, float] = field(default_factory=dict)
    relation_mult: dict[str, float] = field(default_factory=dict)
    node_prior: dict[str, float] = field(default_factory=dict)
    directives: list[dict] = field(default_factory=list)
    action_priors: dict[str, float] = field(default_factory=dict)
    portfolios: list[dict] = field(default_factory=list)
    focus_targets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "cold"
    recalled: int = 0

    def multiplier(self, edge) -> float:
        return (self.edge_mult.get(edge.key, 1.0)
                * self.relation_mult.get(edge.rel, 1.0))


def simulate(estate: Estate, patient_zero, beliefs: Beliefs | None = None,
             trials: int = 2400, horizon: int = 90, seed: int = 7,
             policy: str = "balanced") -> dict:
    beliefs = beliefs or Beliefs()
    nodes = list(estate.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    edges = estate.edges
    E = len(edges)
    rng = np.random.default_rng(seed)

    src = np.array([idx[e.src] for e in edges], dtype=np.int32)
    dst = np.array([idx[e.dst] for e in edges], dtype=np.int32)
    p = np.array([e.p for e in edges], dtype=np.float64)
    t = np.array([e.t for e in edges], dtype=np.float64)
    for i, e in enumerate(edges):
        p[i] *= beliefs.multiplier(e)
        p[i] *= beliefs.node_prior.get(e.src, 1.0) ** 0.35
    # attacker policy shapes both speed and stealth
    if policy == "loud":
        p = np.clip(p * 1.18, 0.02, 0.985)
    elif policy == "quiet":
        p = np.clip(p * 0.80, 0.02, 0.95)
    p = np.clip(p, 0.015, 0.985)

    if isinstance(patient_zero, (list, tuple, set)):
        pool = [idx[x] for x in patient_zero]
    else:
        pool = [idx[patient_zero]]
    p0_idx = rng.choice(pool, size=trials).astype(np.int32)

    attempts = rng.geometric(p, size=(trials, E)).astype(np.float32)
    wait = np.minimum(attempts * t.astype(np.float32), horizon + 2.0)

    dwell_mean = np.array([KIND_DWELL[estate.nodes[n].kind] for n in nodes])
    if policy == "quiet":
        dwell_mean = dwell_mean * 1.45
    elif policy == "loud":
        dwell_mean = dwell_mean * 0.72
    dwell = rng.exponential(dwell_mean, size=(trials, N)).astype(np.float32)
    dwell = np.clip(dwell, 3.0, horizon * 1.6)

    INF = np.float32(1e8)
    t_inf = np.full((trials, N), INF, dtype=np.float32)
    t_inf[np.arange(trials), p0_idx] = 0.0
    parent = np.full((trials, N), -1, dtype=np.int32)

    # vectorised relaxation; at most N passes, usually 6-8
    for _ in range(24):
        changed = False
        for e in range(E):
            s, d = src[e], dst[e]
            base = t_inf[:, s]
            live = base < INF
            if not live.any():
                continue
            cand = base + wait[:, e]
            valid = live & (cand <= horizon) & (cand < base + dwell[:, s])
            better = valid & (cand < t_inf[:, d] - 1e-6)
            if better.any():
                t_inf[better, d] = cand[better]
                parent[better, d] = e
                changed = True
        if not changed:
            break

    infected = t_inf < INF
    critical_idx = [idx[n] for n in estate.critical_ids]
    asset_by_idx = {idx[a.node_id]: a for a in estate.assets}
    # precompute conditional loss per asset at 6h containment
    asset_loss_values = {a.node_id: asset_loss(a, 1.0, 6.0)["total"]
                         for a in estate.assets}
    ai_idx = [idx[n] for n, node in estate.nodes.items() if node.kind == "ai_endpoint"]
    ai_rates = {idx[n]: node.meta.get("abuse_rate", 0.0)
                for n, node in estate.nodes.items() if node.kind == "ai_endpoint"}

    trial_loss = np.zeros(trials)
    reached_assets = np.zeros(trials, dtype=np.int32)
    for ai in critical_idx:
        hit = infected[:, ai]
        reached_assets += hit.astype(np.int32)
        if hit.any():
            trial_loss[hit] += asset_loss_values[nodes[ai]]
    for ai in ai_idx:
        hit = infected[:, ai]
        if hit.any():
            tt = np.where(hit, t_inf[:, ai], 0)
            hrs = np.clip(6.0 - tt / 60.0, 0.15, 6.0)
            trial_loss[hit] += ai_rates[ai] * hrs[hit]

    cred_idx = [idx[c] for c in estate.credential_ids]
    secondary = infected[:, cred_idx].sum(axis=1) - 1
    r0 = float(np.clip(secondary, 0, None).mean())

    ttc = t_inf[:, critical_idx]
    reach_any = (ttc < INF).any(axis=1)
    ttc_min = np.where(reach_any, ttc.min(axis=1), np.nan)
    finite_ttc = ttc_min[np.isfinite(ttc_min)]

    # ---- extract dominant attack paths -----------------------------------
    path_stats: dict[tuple, dict] = {}
    influence = np.zeros(N)
    fired = np.zeros(E)
    fired_count = np.zeros(E, dtype=np.int64)
    for trial in range(trials):
        chain_mask = parent[trial] >= 0
        if chain_mask.any():
            fired_count[np.unique(parent[trial][chain_mask])] += 1
        if not reach_any[trial]:
            continue
        candidates = [(asset_by_idx[a].value, a) for a in critical_idx
                      if t_inf[trial, a] < INF]
        candidates.sort(reverse=True)
        for _, target in candidates[:2]:
            chain_edges: list[int] = []
            chain_nodes: list[str] = []
            cur = target
            guard = 0
            while cur != p0_idx[trial] and guard < 20:
                e = parent[trial, cur]
                if e < 0:
                    break
                chain_edges.append(int(e))
                chain_nodes.append(nodes[cur])
                cur = src[e]
                guard += 1
            if cur != p0_idx[trial]:
                continue
            chain_nodes.append(nodes[cur])
            chain_edges.reverse()
            chain_nodes.reverse()
            key = tuple(chain_edges)
            rec = path_stats.setdefault(key, {"count": 0, "loss": 0.0,
                                              "time": 0.0, "nodes": chain_nodes})
            rec["count"] += 1
            rec["loss"] += float(trial_loss[trial])
            rec["time"] += float(t_inf[trial, target])
            w = float(trial_loss[trial]) / trials
            for nid in chain_nodes:
                influence[idx[nid]] += w

    paths = []
    for key, rec in path_stats.items():
        prob = rec["count"] / trials
        conditional = rec["loss"] / max(1, rec["count"])
        paths.append({
            "id": "pth-" + str(abs(hash(key)) % 10_000_000),
            "edges": [edges[e].key for e in key],
            "nodes": rec["nodes"],
            "hops": max(0, len(rec["nodes"]) - 1),
            "prob": round(prob, 4),
            "expectedLoss": round(conditional * prob),
            "conditionalLoss": round(conditional),
            "minutes": round(rec["time"] / max(1, rec["count"]), 1),
            "labels": [estate.nodes[n].label for n in rec["nodes"]],
            "rels": [edges[e].rel for e in key],
        })
    paths.sort(key=lambda x: -x["expectedLoss"])
    # de-duplicate paths that differ only in edge order
    seen, unique = set(), []
    for path in paths:
        sig = (tuple(path["nodes"]), path["conditionalLoss"] // 10_000)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(path)
        if len(unique) >= 14:
            break
    paths = unique

    node_prob = infected.mean(axis=0)

    # ---- residual paths --------------------------------------------------
    # Rotating patient zero cannot undo access the attacker already obtained.
    # Planning therefore runs on paths that start from every credential the
    # outbreak has plausibly already reached, discounted by how likely that
    # foothold is.  This is why a single rotation is not a containment plan.
    residual = _residual_paths(estate, nodes, idx, node_prob, beliefs,
                               horizon, asset_loss_values)
    paths_for_plan = residual if residual else paths

    curve = []
    for minute in range(0, horizon + 1, 3):
        alive = (t_inf[:, cred_idx] <= minute).sum(axis=1)
        curve.append({"t": minute, "creds": round(float(alive.mean()), 2)})

    hit_count = infected.sum(axis=0)
    hit_sum = np.where(infected, t_inf, 0.0).sum(axis=0)
    node_mean_time = hit_sum / np.maximum(hit_count, 1)

    loss_sorted = np.sort(trial_loss)
    def q(x: float) -> float:
        return float(np.quantile(loss_sorted, x))
    reached_any = int((trial_loss > 0).sum())
    metered = 0.0
    for ai in ai_idx:
        metered += ai_rates[ai] * min(6.0, horizon / 60)
    for a in estate.assets:
        if a.node_id in estate.critical_ids:
            metered += 0

    return {
        "ok": True,
        "seed": seed,
        "trials": trials,
        "horizon": horizon,
        "policy": policy,
        "patientZero": estate.nodes[patient_zero if isinstance(patient_zero, str)
                                    else list(patient_zero)[0]].label,
        "patientZeroId": patient_zero if isinstance(patient_zero, str)
                         else list(patient_zero)[0],
        "beliefs": {
            "source": beliefs.source,
            "recalled": beliefs.recalled,
            "edgeOverrides": len(beliefs.edge_mult),
            "relationOverrides": len(beliefs.relation_mult),
            "directives": len(beliefs.directives),
            "notes": beliefs.notes[:6],
        },
        "kpis": {
            "r0": round(r0, 2),
            "reachProbability": round(float(reach_any.mean()), 3),
            "ttcP50": round(float(np.quantile(finite_ttc, 0.5)), 1) if len(finite_ttc) else None,
            "ttcP10": round(float(np.quantile(finite_ttc, 0.1)), 1) if len(finite_ttc) else None,
            "lossP50": round(q(0.5)),
            "lossP90": round(q(0.9)),
            "lossP99": round(q(0.99)),
            "expectedLoss": round(float(trial_loss.mean())),
            "meanAssets": round(float(reached_assets.mean()), 2),
            "credentialSpread": round(float(secondary.clip(0).mean()), 2),
            "meteredRate": round(metered, 0),
        },
        "curve": curve,
        "paths": paths,
        "residualPaths": residual,
        "planPaths": paths_for_plan,
        "nodeStates": [
            {"id": n, "p": round(float(node_prob[i]), 4),
             "t": round(float(node_mean_time[i]), 1)}
            for i, n in enumerate(nodes)
        ],
        "edgeFire": [
            {"id": e.key, "fire": round(float(fired_count[i] / trials), 3)}
            for i, e in enumerate(edges)
        ],
        "influence": sorted(
            [{"id": nodes[i], "label": estate.nodes[nodes[i]].label,
              "kind": estate.nodes[nodes[i]].kind,
              "value": round(float(influence[i]))}
             for i in range(N) if influence[i] > 0],
            key=lambda x: -x["value"])[:10],
        "superSpreaders": sorted(
            [{"id": n, "label": estate.nodes[n].label, "kind": estate.nodes[n].kind,
              "p": round(float(node_prob[i]), 3),
              "t": round(float(node_mean_time[i]), 1)}
             for i, n in enumerate(nodes)
             if estate.nodes[n].kind in ("credential", "workload", "role")],
            key=lambda x: -(x["p"] * (2 if x["kind"] == "credential" else 1)))[:8],
        "capturedAt": None,
    }


def _residual_paths(estate: Estate, nodes: list[str], idx: dict,
                    node_prob: np.ndarray, beliefs: Beliefs, horizon: int,
                    asset_loss_values: dict) -> list[dict]:
    """Residual exposure from every foothold the outbreak plausibly already has.

    Rotating patient zero cannot undo access the attacker has already obtained.
    For each likely-compromised credential we enumerate the plausible chains to
    every critical asset (up to three distinct routes per asset), weight each
    by foothold probability and keep the highest expected-loss paths.  The
    action portfolio then has to cut a *set* of chains, not one heroic vertex.
    """
    edges = estate.edges
    adj: dict[str, list] = {n: [] for n in nodes}
    for e in edges:
        p = min(0.985, max(0.01, e.p * beliefs.multiplier(e)))
        adj[e.src].append((e.dst, -np.log(p), p, e.t, e))
    for src in adj:
        adj[src].sort(key=lambda x: -x[2])
    asset_ids = {a.node_id: a for a in estate.assets
                 if a.node_id in estate.critical_ids}
    out: list[dict] = []
    for seed in estate.credential_ids:
        sp = float(node_prob[idx[seed]])
        if sp < 0.06:
            continue
        weight = sp * (0.50 if estate.nodes[seed].meta.get("exposed") else 1.0)
        if seed == estate.credential_ids[0]:
            weight *= 0.55

        # enumerate simple chains from this foothold; cap the work per seed
        chains: list[tuple] = []
        stack = [(seed, [seed], [], 0.0, 0.0)]
        budget = 4200
        while stack and budget > 0:
            node, pnodes, pedges, logp, minutes = stack.pop()
            budget -= 1
            if node in asset_ids and pedges:
                chains.append((pnodes, pedges, logp, minutes))
                continue
            if len(pedges) >= 8 or minutes > horizon * 2.2:
                continue
            for dst, _w, p, t, edge in adj.get(node, []):
                if dst in pnodes:
                    continue
                stack.append((dst, pnodes + [dst], pedges + [edge],
                              logp + float(np.log(p)), minutes + t))

        per_asset: dict[str, list[tuple]] = {}
        for chain in chains:
            per_asset.setdefault(chain[0][-1], []).append(chain)
        picked: list[tuple] = []
        for asset_id, options in per_asset.items():
            options.sort(key=lambda c: -c[2])
            seen = set()
            for chain in options:
                sig = tuple(chain[0])
                if sig in seen:
                    continue
                seen.add(sig)
                picked.append(chain)
                if len(seen) >= 5:
                    break

        for path_nodes, path_edges, logp, minutes in picked:
            loss = float(asset_loss_values.get(path_nodes[-1], 0.0))
            if loss <= 0:
                continue
            prob = float(np.exp(logp)) * weight
            if prob < 0.0005:
                continue
            out.append({
                "id": "res-" + str(abs(hash(tuple(e.key for e in path_edges) + (seed,))) % 10_000_000),
                "seed": seed,
                "seedLabel": estate.nodes[seed].label,
                "edges": [e.key for e in path_edges],
                "nodes": path_nodes,
                "labels": [estate.nodes[n].label for n in path_nodes],
                "rels": [e.rel for e in path_edges],
                "prob": round(prob, 5),
                "conditionalLoss": round(loss),
                "expectedLoss": round(loss * prob),
                "minutes": round(minutes, 1),
                "hops": len(path_edges),
            })
    # aggregate identical chains across footholds
    merged: dict[tuple, dict] = {}
    for path in out:
        key = tuple(path["nodes"])
        if key in merged:
            merged[key]["prob"] = round(min(0.99, merged[key]["prob"] + path["prob"]), 5)
            merged[key]["expectedLoss"] = round(
                merged[key]["conditionalLoss"] * merged[key]["prob"])
        else:
            merged[key] = path
    paths = sorted(merged.values(), key=lambda p: -p["expectedLoss"])
    return paths[:64]


def _best_chain(adj, start, target, max_hops: int = 8):
    """Dijkstra over -log(p): the most likely chain, not the shortest hop count."""
    import heapq
    heap = [(0.0, 0.0, start, [], [], 0)]
    best = {start: 0.0}
    while heap:
        cost, minutes, node, path_nodes, path_edges, hops = heapq.heappop(heap)
        if node == target:
            return ([start] + path_nodes, path_edges, -cost, minutes)
        if hops >= max_hops:
            continue
        for dst, w, p, t, edge in adj.get(node, []):
            new_cost = cost + w
            if best.get(dst, 1e9) <= new_cost:
                continue
            best[dst] = new_cost
            heapq.heappush(heap, (new_cost, minutes + t, dst,
                                  path_nodes + [dst], path_edges + [edge], hops + 1))
    return None


def labels(estate: Estate, trials: int = 220, horizon: int = 240,
           seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft supervision for the graph model: P(node reached), P(edge fires)."""
    result = simulate(estate, estate.credential_ids, Beliefs(), trials=trials,
                      horizon=horizon, seed=seed, policy="balanced")
    nodes = list(estate.nodes)
    node_p = np.array([next(s["p"] for s in result["nodeStates"] if s["id"] == n)
                       for n in nodes], dtype=np.float32)
    edge_p = np.array([next(s["fire"] for s in result["edgeFire"] if s["id"] == e.key)
                       for e in estate.edges], dtype=np.float32)
    node_time = np.array([next(s["t"] for s in result["nodeStates"] if s["id"] == n)
                          for n in nodes], dtype=np.float32)
    return node_p, edge_p, node_time
