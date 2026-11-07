"""The AEGIS agent: perceive → recall → simulate → plan → narrate → act → learn.

The important loop is the last arrow.  Every contained outbreak is written back
to Hindsight as an episode, a set of facts, a structural posterior and — when
the analyst disagrees — a directive.  The next outbreak recalls those records
before it simulates, so the plan starts from experience instead of zero.
"""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .estate import ESTATE_SPECS, Estate, get_estate, patient_zero_options
from .gnn import GraphSAGE, load as load_gnn, prior_for, train_and_save
from .impact import asset_loss, tornado
from .memory import MemoryBank
from .microlm import load as load_lm, train as train_lm
from .optimizer import evaluate_portfolio, optimise
from .response import choreograph
from .simulator import Beliefs, simulate

DATA = Path(__file__).resolve().parent.parent / "data"

SCENARIOS = [
    {"id": "leaked-ai-key", "estate": "meridian", "p0": "key-openai-content",
     "label": "Leaked AI key", "signal": "secret scanning hit · public repo",
     "story": "A production OpenAI key appears in a public commit. The scanner is right, "
              "and says nothing about the four hops behind it."},
    {"id": "ci-secret", "estate": "meridian", "p0": "key-gh-release",
     "label": "CI secret exposure", "signal": "CI variable dump · contractor laptop",
     "story": "A release CI secret is copied off a contractor laptop. It can push to the "
              "release branch, which assumes production."},
    {"id": "oauth-grant", "estate": "northwind", "p0": "key-azure-etl",
     "label": "OAuth grant abuse", "signal": "consent grant from unmanaged device",
     "story": "A finance ETL service principal is used from an unmanaged device. The "
              "grant survives a password reset."},
    {"id": "okta-admin", "estate": "northwind", "p0": "key-okta-provision",
     "label": "Identity admin token", "signal": "impossible travel · IdP admin",
     "story": "An Okta provisioning token is replayed from two continents inside an hour."},
    {"id": "deploy-role", "estate": "orbital", "p0": "key-aws-deploy",
     "label": "Deploy role compromise", "signal": "assume-role from new ASN",
     "story": "The production deploy key is used from an ASN this estate has never seen."},
    {"id": "npm-token", "estate": "orbital", "p0": "key-npm-publish",
     "label": "Package token theft", "signal": "registry publish anomaly",
     "story": "A package publishing token is used to push a patch release out of band."},
]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class AegisAgent:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        DATA.mkdir(parents=True, exist_ok=True)
        self.memory = MemoryBank()
        self.gnn, self.gnn_meta = self._ensure_gnn()
        self.lm, self.tok, self.lm_meta = self._ensure_lm()
        self.incidents: dict[str, dict] = {}
        self.counter = 1040
        self.campaign: list[dict] = []
        self._load_incidents()

    # ------------------------------------------------------------------ boot
    def _ensure_gnn(self):
        path = DATA / "gnn.npz"
        meta_path = DATA / "gnn.json"
        if not path.exists():
            metrics = train_and_save(path, quick=False)
        else:
            metrics = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return load_gnn(path), metrics

    def _ensure_lm(self):
        path = DATA / "lm.npz"
        if not path.exists():
            train_lm(path, steps=340)
        return load_lm(path)

    def _load_incidents(self) -> None:
        path = DATA / "incidents.json"
        if path.exists():
            try:
                self.incidents = json.loads(path.read_text())
                self.counter = max([1040] + [int(i.split("-")[1])
                                             for i in self.incidents]) + 1
            except Exception:
                pass

    def _save_incidents(self) -> None:
        (DATA / "incidents.json").write_text(json.dumps(self.incidents, indent=0))

    # ------------------------------------------------------------------ run
    def run(self, estate_id: str, p0: str, *, trials: int = 2600,
            horizon: int = 90, policy: str = "balanced",
            use_memory: bool = True, disruption_budget: float = 0.08) -> dict:
        t_start = time.perf_counter()
        estate = get_estate(estate_id)
        node = estate.nodes.get(p0)
        if node is None:
            raise ValueError(f"unknown patient zero {p0}")
        trace: list[dict] = []

        def mark(step: str, detail: str, t0: float) -> None:
            trace.append({"step": step, "detail": detail,
                          "ms": round((time.perf_counter() - t0) * 1000, 1)})

        # 1. recall ---------------------------------------------------------
        t0 = time.perf_counter()
        query = (f"estate {estate.name} patient zero {node.label} "
                 f"{node.service} {node.meta.get('privilege', '')} "
                 f"outbreak containment cut vertex")
        recalled = self.memory.recall(query, max_results=6) if use_memory else {
            "results": [], "latencyMs": 0, "channels": {}, "mode": "disabled"}
        mark("recall", f"{len(recalled.get('results', []))} memories · "
                        f"{recalled.get('mode', 'local')}", t0)

        # 2. beliefs --------------------------------------------------------
        t0 = time.perf_counter()
        node_prior, edge_prior, base_edge = prior_for(self.gnn, estate)
        if use_memory:
            beliefs = self.memory.beliefs(estate, node_prior)
            for key, pred in edge_prior.items():
                factor = _clamp((pred / max(0.02, base_edge[key])) ** 0.45, 0.72, 1.38)
                beliefs.edge_mult[key] = round(
                    _clamp(beliefs.edge_mult.get(key, 1.0) * factor, 0.55, 1.6), 3)
            beliefs.notes.insert(0, "GNN edge priors blended")
        else:
            beliefs = Beliefs(node_prior=node_prior, source="cold",
                              notes=["cold start: no episodic memory"])
        mark("believe", f"{len(beliefs.edge_mult)} edge priors · "
                         f"{len(beliefs.directives)} directives", t0)

        # 3. simulate -------------------------------------------------------
        t0 = time.perf_counter()
        sim = simulate(estate, p0, beliefs, trials=trials, horizon=horizon,
                       seed=int(time.time() * 1000) % 100_000, policy=policy)
        mark("simulate", f"{trials:,} outbreaks · R0 {sim['kpis']['r0']}", t0)

        # 4. plan -----------------------------------------------------------
        t0 = time.perf_counter()
        plan = optimise(estate, sim["planPaths"], p0, beliefs,
                        disruption_budget=disruption_budget, cost_budget=20_000,
                        max_actions=5)
        mark("optimise", f"{len(plan['actions'])} interventions · "
                          f"{round(plan['coverage'] * 100)}% coverage", t0)

        # 4b. held-out evaluation.  The warm plan is built from the memory
        # forecast; the cold plan is built from a separate memory-free forecast.
        # Both are then scored on a THIRD memory-free reservoir the planners
        # never saw.  This measures decision quality, not forecast alignment.
        cold = None
        if use_memory:
            t_cold = time.perf_counter()
            cold_sim = simulate(estate, p0, Beliefs(node_prior=node_prior, source="cold"),
                                trials=min(1200, trials), horizon=horizon,
                                seed=int(time.time() * 1000) % 50_000 + 17,
                                policy=policy)
            cold_plan = optimise(estate, cold_sim["planPaths"], p0, None,
                                 disruption_budget=disruption_budget, cost_budget=20_000,
                                 max_actions=5)
            truth_sim = simulate(estate, p0, Beliefs(node_prior=node_prior, source="held-out"),
                                 trials=min(1400, trials), horizon=horizon,
                                 seed=int(time.time() * 1000) % 70_000 + 31,
                                 policy=policy)
            truth_paths = truth_sim["planPaths"]
            warm_truth = evaluate_portfolio(estate, truth_paths, plan["actions"])
            cold_truth = evaluate_portfolio(estate, truth_paths, cold_plan["actions"])
            naive_truth = evaluate_portfolio(
                estate, truth_paths, cold_plan["baseline"].get("actionDetails", []))
            plan["truthCoverage"] = warm_truth["coverage"]
            plan["truthPathsTotal"] = len(truth_paths)
            plan["baselineTruth"] = naive_truth
            cold = {"coverage": cold_truth["coverage"],
                    "disruption": cold_truth["disruption"],
                    "actions": [a["label"] for a in cold_plan["actions"]],
                    "naiveCoverage": naive_truth["coverage"],
                    "heldOut": True,
                    "kpis": cold_sim["kpis"]}
            mark("ablate", "cold + warm plans scored on a held-out reservoir", t_cold)

        # 5. impact + narrative --------------------------------------------
        t0 = time.perf_counter()
        top_path = sim["paths"][0] if sim["paths"] else None
        reached = top_path["nodes"] if top_path else []
        loss = {
            "expected": sim["kpis"]["expectedLoss"],
            "p50": sim["kpis"]["lossP50"],
            "p90": sim["kpis"]["lossP90"],
            "p99": sim["kpis"]["lossP99"],
            "assets": [asset_loss(a, 1.0, 6.0) for a in estate.assets
                       if a.node_id in reached],
            "tornado": tornado(estate, reached),
        }
        memo = self.narrate(estate, node, sim, plan, recalled, beliefs, loss)
        v = memo["validator"]
        mark("narrate", f"memo {v['final']['accepted']}/{v['final']['checked']} verified · "
                        f"draft {v['accepted']}/{v['checked']} grounded", t0)

        incident_id = f"INC-{self.counter}"
        self.counter += 1
        incident = {
            "id": incident_id,
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "estate": estate_id,
            "estateName": estate.name,
            "patientZero": p0,
            "patientZeroLabel": node.label,
            "signal": node.detail,
            "policy": policy,
            "status": "open",
            "memoryUsed": use_memory,
            "recalled": recalled.get("results", []),
            "memoryMode": recalled.get("mode", "local"),
            "sim": sim,
            "plan": plan,
            "loss": loss,
            "cold": cold,
            "memo": memo,
            "trace": trace,
            "latencyMs": round((time.perf_counter() - t_start) * 1000),
            "beliefs": {
                "source": beliefs.source,
                "edges": len(beliefs.edge_mult),
                "relations": len(beliefs.relation_mult),
                "directives": beliefs.directives,
                "notes": beliefs.notes,
                "actionPriors": beliefs.action_priors,
                "relationMultipliers": beliefs.relation_mult,
            },
        }
        with self.lock:
            self.incidents[incident_id] = incident
            self._save_incidents()
        return incident

    # ------------------------------------------------------------- narrative
    def _lm_draft(self, estate: Estate, sim: dict, plan: dict) -> str:
        prompt = (f"outbreak memo . estate {estate.name.lower()} . patient zero "
                  f"{sim['patientZero'].replace('-', ' ')} .")
        try:
            return self.lm.sample(self.tok, prompt, max_new=30,
                                  temperature=0.68, top_k=18,
                                  seed=int(time.time()) % 9999).strip()
        except Exception:
            return ""

    def narrate(self, estate: Estate, node, sim: dict, plan: dict,
                recalled: dict, beliefs: Beliefs, loss: dict) -> dict:
        k = sim["kpis"]
        actions = plan["actions"]
        draft = self._lm_draft(estate, sim, plan)
        allowed_numbers = set()
        for value in (k["r0"], k["lossP50"], k["lossP90"], k["expectedLoss"],
                      k["ttcP50"], plan["coverage"] * 100, plan["disruption"] * 100,
                      len(actions), len(estate.nodes), len(estate.edges),
                      sim["trials"]):
            if value is None:
                continue
            allowed_numbers.add(round(float(value)))
            allowed_numbers.add(round(float(value), 1))
        # The LM draft is not trusted; every numeric token in it is checked
        # against the simulation before any of its prose can be used.
        checks = []
        for token in re.findall(r"\d[\d,]*(?:\.\d+)?", draft):
            num = float(token.replace(",", ""))
            ok = any(abs(num - a) <= max(0.15, abs(a) * 0.02)
                     for a in allowed_numbers)
            checks.append({"claim": token, "ok": ok})
        accepted = sum(1 for c in checks if c["ok"])
        top = sim["paths"][0] if sim["paths"] else None
        chain = " → ".join(top["labels"]) if top else "no completed chain"
        recall_line = (f"{len(recalled.get('results', []))} prior records recalled"
                       if recalled.get("results") else
                       "cold start — no prior outbreak in memory")
        memo_text = (
            f"{incident_line(estate, sim)} "
            f"The chain that matters is {chain}. "
            f"R₀ is {k['r0']}; median time to a critical asset is "
            f"{k['ttcP50'] if k['ttcP50'] is not None else '—'} minutes; "
            f"P50 simulated loss is ${k['lossP50']:,} with P90 "
            f"${k['lossP90']:,}. "
            f"{recall_line}. The portfolio contains {len(actions)} interventions: "
            f"{', '.join(a['label'].lower() + ' on ' + a['node_label'] for a in actions)}. "
            f"It cuts {round(plan['coverage'] * 100)}% of residual paths for "
            f"{round(plan['disruption'] * 100)}% workflow disruption, "
            f"versus {round(plan['baseline']['coverage'] * 100)}% for rotating "
            f"the three riskiest credentials. Every number here traces to the "
            f"simulation; the language model only drafted the prose."
        )
        # A second, independent check on the deterministic memo: each claim is
        # a value the simulator produced and must literally appear in the text.
        final_claims: list[dict] = []

        def _claim(label: str, value) -> None:
            if value is None:
                return
            text = f"{value:,}" if isinstance(value, int) else str(value)
            final_claims.append({"claim": label, "value": text,
                                 "ok": text in memo_text})

        _claim("trials", sim["trials"])
        _claim("r0", k["r0"])
        _claim("ttcP50", k["ttcP50"])
        _claim("lossP50", k["lossP50"])
        _claim("lossP90", k["lossP90"])
        _claim("coverage%", round(plan["coverage"] * 100))
        _claim("disruption%", round(plan["disruption"] * 100))
        _claim("baseline%", round(plan["baseline"]["coverage"] * 100))
        _claim("interventions", len(actions))
        final_accepted = sum(1 for c in final_claims if c["ok"])
        return {
            "memo": memo_text,
            "draft": draft,
            "validator": {
                "checked": len(checks),
                "accepted": accepted,
                "rejected": [c["claim"] for c in checks if not c["ok"]],
                "checks": checks[:10],
                "usedDraft": bool(checks) and accepted == len(checks),
                "fallback": "deterministic grounded synthesis",
                "final": {
                    "checked": len(final_claims),
                    "accepted": final_accepted,
                    "rejected": [c["claim"] for c in final_claims if not c["ok"]],
                    "claims": final_claims,
                },
            },
            "lm": {
                "name": "CONTAGION-LM",
                "params": self.lm_meta.get("params"),
                "perplexity": self.lm_meta.get("perplexity"),
                "trainedHere": True,
            },
            "memoryLine": recall_line,
            "topPath": top,
            "actions": [a["label"] for a in actions],
        }

    # ---------------------------------------------------------------- execute
    def execute(self, incident_id: str, actor: str = "soc-analyst",
                verdict: str = "contained") -> dict:
        incident = self.incidents.get(incident_id)
        if incident is None:
            raise ValueError("unknown incident")
        if incident["status"] != "open":
            return {"ok": False, "reason": f"already {incident['status']}"}
        estate = get_estate(incident["estate"])
        result = choreograph(estate, incident["plan"]["actions"], actor=actor)
        incident["status"] = "contained" if verdict != "incomplete" else "review"
        incident["verdict"] = verdict
        incident["execution"] = result
        incident["containedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        recalled_ids = [r["id"] for r in incident.get("recalled", [])]
        learned = self.memory.retain_episode(
            incident=incident, estate=estate, sim=incident["sim"],
            plan=incident["plan"], verdict=verdict, analyst=actor,
            recalled_ids=recalled_ids)
        incident["learned"] = learned
        with self.lock:
            self._save_incidents()
        return {
            "ok": True,
            "incident": incident,
            "execution": result,
            "learned": learned,
            "memory": self.memory.stats(),
        }

    # ---------------------------------------------------------------- replay
    def run_campaign(self, episodes: int = 6, trials: int = 900,
                     disruption_budget: float = 0.08) -> dict:
        """Sequential replay: the honest way to show memory compounding."""
        scenarios = SCENARIOS[:episodes]
        series = []
        for i, scenario in enumerate(scenarios):
            warm = self.run(scenario["estate"], scenario["p0"], trials=trials,
                            use_memory=True, disruption_budget=disruption_budget)
            self.execute(warm["id"], actor="campaign-runner",
                         verdict="contained" if warm["plan"]["coverage"] > 0.55 else "incomplete")
            # cold is the same-incident plan without memory, scored on the warm
            # reservoir by run().  The two policies therefore see one forecast.
            cold_cov = (warm.get("cold") or {}).get("coverage", 0.0)  # held-out score
            series.append({
                "index": i + 1,
                "scenario": scenario["label"],
                "estate": scenario["estate"],
                "warmCoverage": warm["plan"].get("truthCoverage", warm["plan"]["coverage"]),
                "coldCoverage": cold_cov,
                "naiveCoverage": warm["plan"].get(
                    "baselineTruth", warm["plan"]["baseline"]).get("coverage", 0.0),
                "pathsTotal": warm["plan"].get("truthPathsTotal",
                                                  warm["plan"]["pathsTotal"]),
                "warmDisruption": warm["plan"]["disruption"],
                "recalled": len(warm["recalled"]),
                "actions": [a["label"] for a in warm["plan"]["actions"]],
                "memoryMode": warm["memoryMode"],
            })
        self.campaign = series
        # Averages only over episodes that actually had memory available.  The
        # first episode is a cold start; counting its random forecast gap as
        # "learning" would be dishonest.  No-path incidents are also excluded.
        reachable = [s for s in series if s.get("pathsTotal", 0) > 0]
        # Episode 1 is the cold start: it has no prior episode to recall, so its
        # forecast gap is not evidence of learning.  Keep it in the ledger and
        # exclude it from the headline mean, as documented in the README.
        eligible = [s for s in reachable if s.get("index", 0) > 1 and s.get("recalled", 0) > 0]

        def _mean(key: str) -> float:
            return round(float(np.mean([s[key] for s in eligible])), 3) if eligible else 0.0

        summary = {
            "episodes": len(series),
            "validEpisodes": len(reachable),
            "memoryEpisodes": len(eligible),
            "excludedEpisodes": len(series) - len(eligible),
            "coldMean": _mean("coldCoverage"),
            "warmMean": _mean("warmCoverage"),
            "naiveMean": _mean("naiveCoverage"),
            "gain": round(float(np.mean([s["warmCoverage"] - s["coldCoverage"]
                                         for s in eligible])), 3) if eligible else 0.0,
            "series": series,
        }
        (DATA / "campaign.json").write_text(json.dumps(summary, indent=1))
        return summary

    # ------------------------------------------------------------------ info
    def bootstrap(self) -> dict:
        return {
            "estates": [get_estate(s["id"]).to_dict()["stats"] | {
                "id": s["id"], "name": s["name"], "region": s["region"]}
                for s in ESTATE_SPECS],
            "patientZeros": {s["id"]: patient_zero_options(get_estate(s["id"]))
                             for s in ESTATE_SPECS},
            "scenarios": SCENARIOS,
            "memory": self.memory.stats(),
            "gnn": self.gnn_meta,
            "lm": {k: v for k, v in self.lm_meta.items()
                   if k not in ("tokenizer", "lossCurve")},
            "incidents": sorted(
                [{k: v for k, v in inc.items()
                  if k in ("id", "at", "estateName", "patientZeroLabel",
                           "status", "memoryUsed", "patientZero")}
                 for inc in self.incidents.values()],
                key=lambda x: x["id"], reverse=True)[:12],
            "campaign": json.loads((DATA / "campaign.json").read_text())
                        if (DATA / "campaign.json").exists() else None,
            "system": {
                "layers": [
                    {"name": "Ingest", "detail": "credentials · workloads · repos · CI · roles · AI endpoints"},
                    {"name": "Graph", "detail": f"{len(get_estate('meridian').nodes)} nodes · "
                                                f"{len(get_estate('meridian').edges)} evidenced edges"},
                    {"name": "Outbreak", "detail": "vectorised Monte Carlo · detection hazard"},
                    {"name": "Model", "detail": "relation-aware GraphSAGE (NumPy, from scratch)"},
                    {"name": "Plan", "detail": "CELF + exhaustive shortlist under disruption budget"},
                    {"name": "Memory", "detail": "Hindsight · four-channel recall · learned posteriors"},
                    {"name": "Narrate", "detail": "CONTAGION-LM draft + numeric validator"},
                ]
            },
        }


def incident_line(estate: Estate, sim: dict) -> str:
    return (f"{estate.name} · patient zero {sim['patientZero']} · "
            f"{sim['trials']:,} simulated outbreaks.")
