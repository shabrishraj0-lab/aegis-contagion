"""Ring vaccination: a constrained maximum-coverage problem.

Choose a *portfolio* of interventions that removes the most expected reachable
loss per unit of workflow disruption.  Greedy with lazy evaluation (CELF)
builds the frontier; an exhaustive beam search over the shortlisted actions
certifies the final portfolio for the small k used in the console.

The action menu is deliberately wider than "rotate": revoking sessions or an
OAuth grant is often the correct first move, and a rotation-only product
misses it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .estate import Estate

ACTION_META = {
    "deny_asn":       {"label": "Deny source ASN", "minutes": 2, "cost": 0,
                       "disruption": 0.004, "reversible": True, "icon": "◈"},
    "revoke_sessions": {"label": "Revoke active sessions", "minutes": 3, "cost": 150,
                        "disruption": 0.012, "reversible": True, "icon": "⟳"},
    "remove_oauth":   {"label": "Remove OAuth grant", "minutes": 4, "cost": 400,
                       "disruption": 0.022, "reversible": True, "icon": "⊘"},
    "step_up_mfa":    {"label": "Enforce step-up MFA", "minutes": 5, "cost": 0,
                       "disruption": 0.018, "reversible": True, "icon": "⌘"},
    "reduce_scope":   {"label": "Reduce scope", "minutes": 35, "cost": 1200,
                       "disruption": 0.035, "reversible": True, "icon": "⊟"},
    "rotate":         {"label": "Rotate credential", "minutes": 48, "cost": 3400,
                       "disruption": 0.075, "reversible": True, "icon": "↻"},
    "disable":        {"label": "Disable credential", "minutes": 6, "cost": 0,
                       "disruption": 0.11, "reversible": False, "icon": "✕"},
}


@dataclass
class Action:
    id: str
    kind: str
    target: str
    label: str
    node_label: str
    minutes: int
    cost: float
    disruption: float
    reversible: bool
    icon: str
    node_ids: list[str] = field(default_factory=list)
    paths_blocked: int = 0
    expected_blocked: float = 0.0
    conditional_blocked: float = 0.0
    coverage: float = 0.0
    rationale: str = ""


def _path_survival(action: Action, path: dict, estate: Estate,
                   edge_map: dict) -> float:
    """Residual survival probability of one path after one action."""
    nodes = path["nodes"]
    if action.kind in ("rotate", "disable"):
        return 0.0 if action.target in nodes else 1.0
    if action.kind == "revoke_sessions":
        return 0.12 if action.target in nodes else 1.0
    if action.kind == "remove_oauth":
        if action.target not in nodes:
            return 1.0
        idx = nodes.index(action.target)
        if idx < len(path["edges"]):
            rel = edge_map[path["edges"][idx]].rel
            if rel in ("authenticates", "trusts", "can_invoke", "push_access"):
                return 0.0
        return 0.55
    if action.kind == "step_up_mfa":
        return 0.38 if action.target in nodes else 1.0
    if action.kind == "reduce_scope":
        return 0.45 if action.target in nodes else 1.0
    if action.kind == "deny_asn":
        return 0.30 if nodes and nodes[0] == action.target else 1.0
    return 1.0


def _coverage(actions: list[Action], paths: list[dict], estate: Estate,
              edge_map: dict) -> tuple[float, float, int]:
    base = sum(p["expectedLoss"] for p in paths)
    if base <= 0:
        # no residual chain reaches a critical asset: coverage is undefined,
        # not 100%.  Returning 1.0 here would fabricate a perfect plan.
        return 0.0, 0.0, 0
    remaining = 0.0
    blocked_paths = 0
    for path in paths:
        survival = 1.0
        for action in actions:
            survival *= _path_survival(action, path, estate, edge_map)
        remaining += path["expectedLoss"] * survival
        if survival < 0.2:
            blocked_paths += 1
    return (base - remaining), (base - remaining) / base, blocked_paths


def evaluate_portfolio(estate: Estate, paths: list[dict],
                       actions: list[dict] | list[Action]) -> dict:
    """Score an existing portfolio on a path reservoir.  Used to compare a
    cold-start plan and a memory-conditioned plan on the *same* forecast."""
    edge_map = {e.key: e for e in estate.edges}
    acts: list[Action] = []
    for a in actions:
        if isinstance(a, Action):
            acts.append(a)
            continue
        meta = ACTION_META[a["kind"]]
        node = estate.nodes[a["target"]]
        acts.append(Action(
            id=a.get("id", f"{a['kind']}:{a['target']}"), kind=a["kind"],
            target=a["target"], label=a.get("label", meta["label"]),
            node_label=a.get("node_label", node.label), minutes=meta["minutes"],
            cost=meta["cost"], disruption=a.get("disruption", meta["disruption"]),
            reversible=meta["reversible"], icon=meta["icon"],
            node_ids=[a["target"]]))
    blocked, cov, count = _coverage(acts, paths, estate, edge_map)
    return {
        "coverage": round(cov, 4), "blocked": round(blocked),
        "pathsBlocked": count, "pathsTotal": len(paths),
        "disruption": round(sum(a.disruption for a in acts), 3),
        "cost": round(sum(a.cost for a in acts)),
        "actions": [a.label for a in acts],
    }


def _build_actions(estate: Estate, paths: list[dict], p0: str,
                   beliefs) -> list[Action]:
    edge_map = {e.key: e for e in estate.edges}
    candidates: dict[str, Action] = {}
    node_paths: dict[str, int] = {}
    for path in paths:
        for nid in path["nodes"]:
            node_paths[nid] = node_paths.get(nid, 0) + 1

    def add(kind: str, target: str, rationale: str, *, extra_delay=False) -> None:
        meta = ACTION_META[kind]
        node = estate.nodes[target]
        key = f"{kind}:{target}"
        if key in candidates:
            return
        disruption = meta["disruption"]
        # learned analyst preference lowers operational friction
        if beliefs:
            pref = beliefs.action_priors.get(kind)
            if pref:
                # Procedural memory: operators learned the real cost of this
                # action is lower than the generic playbook estimate.
                disruption *= max(0.40, 1.0 - 0.45 * pref)
        candidates[key] = Action(
            id=key, kind=kind, target=target, label=meta["label"],
            node_label=node.label, minutes=meta["minutes"] + (15 if extra_delay else 0),
            cost=meta["cost"], disruption=round(disruption, 3),
            reversible=meta["reversible"], icon=meta["icon"],
            node_ids=[target], rationale=rationale)

    # fast, cheap interventions first: enough to cover the whole estate
    add("deny_asn", p0, "Cuts the observed external access path without touching production.")
    for nid in node_paths:
        node = estate.nodes[nid]
        if node.kind == "credential":
            add("revoke_sessions", nid, "Invalidates live sessions; leaves the secret valid.")
            add("step_up_mfa", nid, "Bearer reuse fails without a second factor.")
            add("rotate", nid, "Removes the secret itself; the only durable cut.")
            if node.service in ("github", "okta", "gitlab", "npm"):
                add("remove_oauth", nid, "Kills delegated access that rotation leaves intact.")
            add("reduce_scope", nid, "Shrinks reachable assets without an outage.")
        elif node.kind == "role":
            add("revoke_sessions", nid, "Terminates assumed-role sessions across the estate.")
            add("reduce_scope", nid, "Removes the data-plane permissions the chain needs.")
        elif node.kind == "workload":
            add("step_up_mfa", nid, "Forces re-attestation before the workload can act.")
    return list(candidates.values())


def optimise(estate: Estate, paths: list[dict], p0: str, beliefs=None,
             disruption_budget: float = 0.12, cost_budget: float = 20_000,
             max_actions: int = 4) -> dict:
    edge_map = {e.key: e for e in estate.edges}
    actions = _build_actions(estate, paths, p0, beliefs)
    base = sum(p["expectedLoss"] for p in paths) or 1.0

    # directives from memory
    blocked_nodes: set[str] = set()
    applied: list[dict] = []
    if beliefs:
        for d in beliefs.directives:
            if d.get("kind") == "avoid_rotation":
                blocked_nodes.add(d.get("target", ""))
                applied.append({"directive": d.get("text", ""), "effect": "rotation excluded"})
            elif d.get("kind") == "prefer_action":
                applied.append({"directive": d.get("text", ""),
                                "effect": f"{d.get('target')} friction reduced"})
            elif d.get("kind") == "disruption_cap":
                disruption_budget = min(disruption_budget, float(d.get("value", 0.12)))
                applied.append({"directive": d.get("text", ""), "effect": "budget tightened"})
    if blocked_nodes:
        actions = [a for a in actions if not (a.kind == "rotate" and a.target in blocked_nodes)]

    # score each action alone
    for action in actions:
        blocked, cov, count = _coverage([action], paths, estate, edge_map)
        action.expected_blocked = blocked
        action.coverage = cov
        action.paths_blocked = count
        action.conditional_blocked = (
            blocked / max(1e-9, sum(p["expectedLoss"] for p in paths
                                    if action.target in p["nodes"])) * 1.0)

    # ---- greedy with lazy evaluation (CELF) ------------------------------
    selected: list[Action] = []
    remaining = list(actions)
    total_disruption = 0.0
    total_cost = 0.0
    current_blocked = 0.0
    frontier = []
    for step in range(max_actions):
        best = None
        best_gain = -1.0
        for action in remaining:
            trial = selected + [action]
            if sum(a.disruption for a in trial) > disruption_budget + 1e-9:
                continue
            if sum(a.cost for a in trial) > cost_budget:
                continue
            blocked, cov, _ = _coverage(trial, paths, estate, edge_map)
            gain = blocked - current_blocked
            if gain > best_gain:
                best_gain, best = gain, (action, blocked, cov)
        if best is None or best_gain <= 0:
            break
        action, blocked, cov = best
        selected.append(action)
        remaining.remove(action)
        current_blocked = blocked
        total_disruption += action.disruption
        total_cost += action.cost
        frontier.append({
            "k": len(selected), "coverage": round(cov, 4),
            "blocked": round(blocked), "disruption": round(total_disruption, 3),
            "cost": round(total_cost),
            "action": action.label, "target": action.node_label,
        })

    # ---- exhaustive beam refinement over the shortlist --------------------
    # Memory does not just change probabilities: it tells the search where to
    # look.  Targets that were choke points in recalled episodes are promoted
    # into the shortlist, and whole recalled portfolios are re-evaluated.
    shortlist = sorted(actions, key=lambda a: -a.expected_blocked)[:12]
    focus_targets: list[str] = []
    if beliefs:
        focus_targets = list(dict.fromkeys(getattr(beliefs, "focus_targets", []) or []))
        extra = [a for a in sorted(actions, key=lambda a: -a.expected_blocked)
                 if a.target in focus_targets and a not in shortlist]
        shortlist = shortlist + extra[:8]
    best_set: tuple[Action, ...] = tuple(selected)
    best_score = _coverage(selected, paths, estate, edge_map)[0] if selected else 0.0
    best_disc = sum(a.disruption for a in selected)
    seeded_from = None
    for size in range(1, max_actions + 1):
        for combo in itertools.combinations(shortlist, size):
            disc = sum(a.disruption for a in combo)
            cost = sum(a.cost for a in combo)
            if disc > disruption_budget + 1e-9 or cost > cost_budget:
                continue
            blocked, cov, _ = _coverage(list(combo), paths, estate, edge_map)
            # prefer coverage, then lower disruption
            score = blocked - disc * base * 0.15
            if score > best_score + base * 0.001:
                best_score = score
                best_set = combo
                best_disc = disc
    # case-based memory: replay portfolios that worked on this estate before
    if beliefs and getattr(beliefs, "portfolios", None):
        by_key = {(a.kind, a.target): a for a in actions}
        for seed in beliefs.portfolios[-10:]:
            combo: list[Action] = []
            usable = True
            for kind, target in zip(seed.get("actions", []), seed.get("targets", [])):
                action = by_key.get((kind, target))
                if action is None:
                    usable = False
                    break
                if action not in combo:
                    combo.append(action)
            if not usable or not combo:
                continue
            disc = sum(a.disruption for a in combo)
            cost = sum(a.cost for a in combo)
            if disc > disruption_budget + 1e-9 or cost > cost_budget:
                continue
            blocked, cov, _ = _coverage(combo, paths, estate, edge_map)
            score = blocked - disc * base * 0.15
            if score > best_score + base * 0.001:
                best_score = score
                best_set = tuple(combo)
                best_disc = disc
                seeded_from = seed.get("incident")
    if not best_set:
        best_set = tuple(selected[:1])
    chosen = list(best_set)

    # ---- order: reversible, fast actions first; rotations last -----------
    chosen.sort(key=lambda a: (a.kind in ("rotate",), a.minutes))
    blocked, cov, count = _coverage(chosen, paths, estate, edge_map)
    total_disc = sum(a.disruption for a in chosen)
    total_cost = sum(a.cost for a in chosen)
    for action in chosen:
        blocked_one, cov_one, _ = _coverage([action], paths, estate, edge_map)
        action.expected_blocked = round(blocked_one)
        action.coverage = round(cov_one, 4)
        action.rationale = _rationale(action, estate)

    # ---- naive baseline: rotate the three most "risky" credentials -------
    risky = sorted(
        [n for n in estate.credential_ids],
        key=lambda c: -(len(estate.out[c]) * (2 if estate.nodes[c].meta.get("exposure") else 1)),
    )[:3]
    naive = []
    for nid in risky:
        naive.append(Action(
            id=f"rotate:{nid}", kind="rotate", target=nid,
            label="Rotate credential", node_label=estate.nodes[nid].label,
            minutes=48, cost=3400, disruption=0.075, reversible=True, icon="↻",
            node_ids=[nid]))
    naive_blocked, naive_cov, naive_count = _coverage(naive, paths, estate, edge_map)

    # ---- frontier of best coverage at each disruption budget -------------
    budget_curve = []
    for budget in (0.02, 0.04, 0.06, 0.09, 0.12, 0.18, 0.26):
        picks: list[Action] = []
        remaining2 = list(actions)
        blocked_prev = 0.0
        while len(picks) < max_actions:
            best2 = None
            gain2 = 1e-9
            for action in remaining2:
                if sum(a.disruption for a in picks + [action]) > budget:
                    continue
                b2, _, _ = _coverage(picks + [action], paths, estate, edge_map)
                if b2 - blocked_prev > gain2:
                    gain2 = b2 - blocked_prev
                    best2 = action
            if best2 is None:
                break
            picks.append(best2)
            remaining2.remove(best2)
            blocked_prev = _coverage(picks, paths, estate, edge_map)[0]
        budget_curve.append({
            "budget": budget,
            "coverage": round(_coverage(picks, paths, estate, edge_map)[1], 3) if picks else 0.0,
            "actions": len(picks),
            "disruption": round(sum(a.disruption for a in picks), 3),
        })

    return {
        "coverage": round(cov, 4),
        "blockedExpected": round(blocked),
        "blockedConditional": round(blocked),
        "pathsBlocked": count,
        "pathsTotal": len(paths),
        "disruption": round(total_disc, 3),
        "cost": round(total_cost),
        "actions": [
            {**a.__dict__, "coverage": round(a.coverage, 4),
             "expectedBlocked": round(a.expected_blocked)}
            for a in chosen
        ],
        "frontier": frontier,
        "budgetCurve": budget_curve,
        "baseline": {
            "coverage": round(naive_cov, 4),
            "disruption": round(sum(a.disruption for a in naive), 3),
            "actions": [f"rotate {a.node_label}" for a in naive],
            "actionDetails": [
                {"kind": a.kind, "target": a.target, "label": a.label,
                 "node_label": a.node_label, "disruption": a.disruption}
                for a in naive
            ],
            "pathsBlocked": naive_count,
            "blockedExpected": round(naive_blocked),
        },
        "directivesApplied": applied,
        "seededFrom": seeded_from,
        "focusTargets": len(focus_targets),
        "search": {"candidates": len(actions),
                   "shortlist": len(shortlist),
                   "combinationsEvaluated": sum(
                       len(list(itertools.combinations(shortlist, k)))
                       for k in range(1, max_actions + 1))},
    }


def _rationale(action: Action, estate: Estate) -> str:
    target = estate.nodes[action.target]
    if action.kind == "deny_asn":
        return "Cheap externality cut: removes the observed source without touching production."
    if action.kind == "revoke_sessions":
        return f"Terminates live sessions for {target.label}; the secret itself stays valid."
    if action.kind == "remove_oauth":
        return "Delegated access often survives rotation; removing the grant closes the chain."
    if action.kind == "step_up_mfa":
        return "A stolen bearer token fails without the second factor."
    if action.kind == "reduce_scope":
        return "Removes the data-plane permissions the path needs, without an outage."
    if action.kind == "rotate":
        return "Durable cut: the credential on this path is replaced and the old one revoked."
    if action.kind == "disable":
        return "Immediate but irreversible; reserved for credentials with no live consumers."
    return ""
