"""Hindsight memory: episodic, semantic, structural and procedural.

Three layers cooperate:

1. **Hindsight** (cloud or local mirror) stores the narrative record — what
   happened, what was tried, what the analyst decided and why.  When
   HINDSIGHT_BASE_URL is configured the official `hindsight-client` is used;
   otherwise a local implementation with the same four-channel retrieval
   semantics runs, so the demo never depends on venue Wi-Fi.

2. **Structure posteriors** turn past outbreaks into numeric priors: how often
   each relation and edge actually transmitted, and what each intervention
   bought per unit of disruption.  Those priors are what change the next plan.

3. **A learned reranker** (MemoryRank, a small NumPy MLP) scores recalled
   memories by usefulness.  Analyst verdicts update it online — memory that
   changed a decision for the better is recalled harder next time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .estate import Estate
from .simulator import Beliefs

DATA = Path(__file__).resolve().parent.parent / "data"
BANK_FILE = DATA / "memory_bank.json"

STOP = set("""a an and are as at be by for from has have how in into is it its
of on or that the their then there these this to was were what when where which
who why will with not no but if than them they we you our""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_.:\-]+", text.lower()) if t not in STOP]


def _trigrams(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text.lower())
    return [clean[i:i + 3] for i in range(max(0, len(clean) - 2))]


class Embedder:
    """Deterministic hashed n-gram embedder — no model download, no network."""
    dim = 384

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}

    def vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        grams = _tokens(text) + _trigrams(text)
        for gram in grams:
            h = int(hashlib.blake2b(gram.encode(), digest_size=8).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            weight = self.idf.get(gram, 1.0)
            vec[idx] += sign * weight
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


class MemoryRank:
    """Small learned reranker: 6 features -> 8 hidden -> 1. Online SGD."""

    def __init__(self, seed: int = 12) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((6, 8)).astype(np.float32) * 0.4
        self.b1 = np.zeros(8, dtype=np.float32)
        self.w2 = rng.standard_normal(8).astype(np.float32) * 0.3
        self.b2 = np.float32(0.0)
        self.trained = 0

    def score(self, feats: np.ndarray) -> np.ndarray:
        h = np.maximum(feats @ self.W1 + self.b1, 0)
        return 1 / (1 + np.exp(-(h @ self.w2 + self.b2)))

    def update(self, feats: np.ndarray, target: np.ndarray, lr: float = 0.05) -> None:
        h = np.maximum(feats @ self.W1 + self.b1, 0)
        out = 1 / (1 + np.exp(-(h @ self.w2 + self.b2)))
        d = (out - target)[:, None]
        grad_w2 = (d * h).mean(axis=0)
        grad_b2 = float(d.mean())
        dh = d * self.w2[None, :] * (h > 0)
        grad_W1 = feats.T @ dh / len(feats)
        grad_b1 = dh.mean(axis=0)
        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.trained += len(feats)

    def state(self) -> dict:
        return {"W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": float(self.b2),
                "trained": self.trained}

    def load(self, state: dict) -> None:
        self.W1 = np.array(state["W1"], dtype=np.float32)
        self.b1 = np.array(state["b1"], dtype=np.float32)
        self.w2 = np.array(state["w2"], dtype=np.float32)
        self.b2 = np.float32(state["b2"])
        self.trained = int(state.get("trained", 0))


class LocalHindsight:
    """Same four-channel recall semantics as Hindsight, implemented locally.

    dense (semantic) · keyword (BM25) · graph (entity expansion) · temporal
    fused with reciprocal-rank fusion, then reranked.
    """
    backend = "local-hindsight"

    def __init__(self, path: Path = BANK_FILE) -> None:
        self.path = path
        self.embedder = Embedder()
        self.records: list[dict] = []
        self.docs: dict[str, Counter] = {}
        self.df: Counter = Counter()
        self.entities: dict[str, set[str]] = {}
        self.reranker = MemoryRank()
        self._last_feats: dict[str, np.ndarray] = {}
        self._load()
        self._rebuild()

    # ------------------------------------------------------------------ bank
    def create_bank(self, bank_id: str, name: str | None = None, **kw) -> dict:
        return {"bank_id": bank_id, "name": name or bank_id, "ok": True}

    def _rebuild(self) -> None:
        self.docs = {}
        self.df = Counter()
        self.entities = {}
        for rec in self.records:
            toks = _tokens(rec["text"])
            self.docs[rec["id"]] = Counter(toks)
            for tok in set(toks):
                self.df[tok] += 1
            for ent in rec.get("entities", []):
                self.entities.setdefault(ent.lower(), set()).add(rec["id"])

    # ---------------------------------------------------------------- retain
    def retain(self, bank_id: str, content: str, metadata: dict | None = None,
               tags: list[str] | None = None, context: str | None = None,
               **kw) -> dict:
        metadata = metadata or {}
        rec = {
            "id": "mem-" + hashlib.sha1(
                (content + str(time.time_ns())).encode()).hexdigest()[:10],
            "text": content,
            "type": metadata.get("type", "observation"),
            "entities": [e.strip() for e in metadata.get("entities", "").split(",") if e.strip()],
            "tags": tags or [],
            "context": context or "",
            "meta": metadata,
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "utility": float(metadata.get("utility", 0.5)),
            "uses": 0,
            "vec": self.embedder.vector(content).tolist(),
        }
        # semantic dedupe: the same fact repeated reinforces rather than bloats
        best, best_sim = None, 0.0
        q = np.array(rec["vec"], dtype=np.float32)
        for other in self.records[-60:]:
            sim = float(np.dot(q, np.array(other["vec"], dtype=np.float32)))
            if sim > best_sim:
                best, best_sim = other, sim
        if best is not None and best_sim > 0.94 and best["type"] == rec["type"]:
            best["utility"] = min(1.0, best["utility"] + 0.02)
            best["meta"]["reinforcements"] = int(best["meta"].get("reinforcements", 1)) + 1
            self._save()
            return {"id": best["id"], "deduped": True, "similarity": round(best_sim, 3)}
        self.records.append(rec)
        self._rebuild()
        self._save()
        return {"id": rec["id"], "deduped": False}

    def list_memories(self, bank_id: str, limit: int = 200) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "vec"}
                for r in self.records[-limit:]]

    # ---------------------------------------------------------------- recall
    def _bm25(self, query: str, rec: dict) -> float:
        toks = _tokens(query)
        doc = self.docs.get(rec["id"], Counter())
        if not doc or not toks:
            return 0.0
        n = max(1, len(self.records))
        avg = sum(sum(d.values()) for d in self.docs.values()) / n
        dl = sum(doc.values())
        k1, b = 1.5, 0.75
        score = 0.0
        for tok in toks:
            tf = doc.get(tok, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - self.df[tok] + 0.5) / (self.df[tok] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avg)))
        return score

    def _channels(self, query: str, types: list[str] | None) -> tuple[dict, dict]:
        pool = [r for r in self.records if not types or r["type"] in types]
        if not pool:
            return {}, {}
        qv = self.embedder.vector(query)
        q_tokens = set(_tokens(query))
        q_entities = {e.lower() for e in _tokens(query) if e in self.entities}
        now = datetime.now(timezone.utc)

        dense, keyword, graph, temporal = {}, {}, {}, {}
        for rec in pool:
            dense[rec["id"]] = float(np.dot(qv, np.array(rec["vec"], dtype=np.float32)))
            bm = self._bm25(query, rec)
            keyword[rec["id"]] = bm
            ent_overlap = len(set(e.lower() for e in rec.get("entities", [])) & q_entities)
            if ent_overlap == 0:
                expansions = set()
                for ent in q_entities:
                    for mid in self.entities.get(ent, set()):
                        expansions |= set(e.lower() for e in
                                          next((x for x in pool if x["id"] == mid), {})
                                          .get("entities", []))
                ent_overlap = len(set(e.lower() for e in rec.get("entities", [])) & expansions) * 0.5
            graph[rec["id"]] = float(ent_overlap)
            try:
                age_h = (now - datetime.fromisoformat(rec["ts"].replace("Z", "+00:00"))
                         ).total_seconds() / 3600
            except Exception:
                age_h = 0.0
            temporal[rec["id"]] = 1.0 / (1.0 + age_h / 24.0)

        channels = {"dense": dense, "keyword": keyword,
                    "graph": graph, "temporal": temporal}
        # reciprocal rank fusion
        fused: dict[str, float] = {}
        for name, scores in channels.items():
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            for rank, (mid, _) in enumerate(ranked[:40]):
                weight = {"dense": 1.0, "keyword": 0.9, "graph": 0.7,
                          "temporal": 0.35}[name]
                fused[mid] = fused.get(mid, 0.0) + weight / (55 + rank)
        return fused, channels

    def recall(self, bank_id: str, query: str, types: list[str] | None = None,
               max_results: int = 6, budget: str = "mid", **kw) -> dict:
        t0 = time.perf_counter()
        fused, channels = self._channels(query, types)
        if not fused:
            return {"results": [], "latencyMs": 0, "channels": {}}
        ids = sorted(fused, key=lambda k: -fused[k])[:24]
        recs = {r["id"]: r for r in self.records if r["id"] in ids}
        qv = self.embedder.vector(query)
        feats, targets = [], []
        for mid in ids:
            rec = recs[mid]
            feats.append([
                float(np.dot(qv, np.array(rec["vec"], dtype=np.float32))),
                min(1.0, channels["keyword"].get(mid, 0.0) / 8.0),
                min(1.0, channels["graph"].get(mid, 0.0) / 3.0),
                channels["temporal"].get(mid, 0.0),
                1.0 if rec["type"] == "episode" else 0.6,
                rec.get("utility", 0.5),
            ])
            targets.append(rec.get("utility", 0.5))
        F = np.array(feats, dtype=np.float32)
        for i, mid in enumerate(ids):
            self._last_feats[mid] = F[i]
        learned = self.reranker.score(F)
        combined = {mid: fused[mid] * 40 + float(learned[i]) * 0.35
                    for i, mid in enumerate(ids)}
        order = sorted(combined, key=lambda k: -combined[k])[:max_results]
        results = []
        for mid in order:
            rec = recs[mid]
            rec["uses"] = int(rec.get("uses", 0)) + 1
            results.append({
                "id": mid, "text": rec["text"], "type": rec["type"],
                "ts": rec["ts"], "entities": rec.get("entities", []),
                "tags": rec.get("tags", []),
                "score": round(min(1.0, combined[mid] / 3.0), 3),
                "utility": round(float(rec.get("utility", 0.5)), 3),
                "channels": {
                    "dense": round(channels["dense"].get(mid, 0.0), 3),
                    "keyword": round(channels["keyword"].get(mid, 0.0), 3),
                    "graph": round(channels["graph"].get(mid, 0.0), 3),
                    "temporal": round(channels["temporal"].get(mid, 0.0), 3),
                },
                "meta": rec.get("meta", {}),
            })
        self._save()
        return {
            "results": results,
            "latencyMs": round((time.perf_counter() - t0) * 1000, 1),
            "candidates": len(ids),
            "backend": self.backend,
            "channels": {k: round(float(np.mean(list(v.values()))), 3)
                         for k, v in channels.items() if v},
        }

    def reflect(self, bank_id: str, query: str, budget: str = "low", **kw) -> dict:
        recall = self.recall(bank_id, query, max_results=8)
        if not recall["results"]:
            return {"text": "No prior outbreaks in memory.", "citations": []}
        episodes = [r for r in recall["results"] if r["type"] == "episode"][:3]
        if not episodes:
            episodes = recall["results"][:3]
        lines = ["Across recalled incidents:"]
        for rec in episodes:
            lines.append("· " + rec["text"])
        return {"text": "\n".join(lines), "citations": [r["id"] for r in episodes]}

    def feedback(self, useful_ids: list[str], good: bool) -> dict:
        """Analyst feedback trains the reranker and the utility prior."""
        if not useful_ids:
            return {"updated": 0}
        recs = [r for r in self.records if r["id"] in useful_ids]
        if not recs:
            return {"updated": 0}
        feats, target = [], []
        for rec in recs:
            rec["utility"] = float(np.clip(
                rec.get("utility", 0.5) + (0.12 if good else -0.10), 0.05, 0.99))
            cached = self._last_feats.get(rec["id"])
            if cached is None:
                continue
            feat = cached.copy()
            feat[5] = rec["utility"]
            feats.append(feat)
            target.append(1.0 if good else 0.0)
        if not feats:
            self._save()
            return {"updated": len(recs)}
        self.reranker.update(np.array(feats, dtype=np.float32),
                             np.array(target, dtype=np.float32))
        self._save()
        return {"updated": len(recs), "trained": self.reranker.trained}

    def stats(self) -> dict:
        counts = Counter(r["type"] for r in self.records)
        return {"records": len(self.records), "byType": dict(counts),
                "backend": self.backend,
                "rerankerParams": int(self.reranker.W1.size + self.reranker.w2.size + 9),
                "rerankerUpdates": self.reranker.trained}

    # ------------------------------------------------------------------ io
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": self.records,
            "reranker": self.reranker.state(),
            "savedAt": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(payload, indent=0))

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text())
            self.records = payload.get("records", [])
            if payload.get("reranker"):
                self.reranker.load(payload["reranker"])
        except Exception:
            self.records = []


class HindsightCloud:
    """Thin wrapper over the official SDK; failures fall back to local memory."""

    def __init__(self, base_url: str, api_key: str, bank_id: str) -> None:
        from hindsight_client import Hindsight
        self.client = Hindsight(base_url=base_url, api_key=api_key, timeout=12.0)
        self.bank_id = bank_id
        self.backend = "hindsight-cloud"
        try:
            self.client.create_bank(bank_id=bank_id, name="AEGIS Contagion")
        except Exception:
            pass

    def retain(self, bank_id, content, metadata=None, tags=None, context=None, **kw):
        meta = {k: str(v) for k, v in (metadata or {}).items()}
        res = self.client.retain(bank_id=bank_id or self.bank_id, content=content,
                                 metadata=meta, tags=tags or [], context=context)
        return {"id": getattr(res, "id", "cloud"), "deduped": False}

    def recall(self, bank_id, query, types=None, max_results=8, **kw):
        res = self.client.recall(bank_id=bank_id or self.bank_id, query=query,
                                 budget="mid")
        results = []
        for item in getattr(res, "results", [])[:max_results]:
            results.append({
                "id": getattr(item, "id", "cloud"),
                "text": getattr(item, "text", str(item)),
                "type": "memory", "score": float(getattr(item, "score", 0.5) or 0.5),
                "channels": {}, "utility": 0.5, "ts": "",
            })
        return {"results": results, "latencyMs": 0, "backend": self.backend,
                "channels": {}}

    def reflect(self, bank_id, query, **kw):
        res = self.client.reflect(bank_id=bank_id or self.bank_id, query=query)
        return {"text": getattr(res, "text", ""), "citations": []}


class MemoryBank:
    """The learning layer the agent actually talks to."""

    def __init__(self, bank_id: str = "aegis-contagion", path: Path = BANK_FILE) -> None:
        self.bank_id = bank_id
        self.local = LocalHindsight(path)
        self.cloud = None
        self.mode = "local"
        base = os.environ.get("HINDSIGHT_BASE_URL", "").strip()
        key = os.environ.get("HINDSIGHT_API_KEY", "").strip()
        if base and key:
            try:
                self.cloud = HindsightCloud(base, key,
                                            os.environ.get("HINDSIGHT_BANK", bank_id))
                self.mode = "cloud+local"
            except Exception as exc:
                self.cloud_error = str(exc)
                self.mode = "local (cloud unavailable)"
        self.structure: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}
        self.action_stats: dict[str, dict] = {}
        self.directives: list[dict] = []
        self.episodes: list[dict] = []
        self._load_numeric()
        self._op_log: list[dict] = []

    # ------------------------------------------------------------- numeric io
    def _num_path(self) -> Path:
        return self.local.path.with_name(f"memory_numeric_{self.bank_id}.json")

    def _load_numeric(self) -> None:
        path = self._num_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text())
            self.structure = payload.get("structure", {})
            self.edges = payload.get("edges", {})
            self.action_stats = payload.get("action_stats", {})
            self.directives = payload.get("directives", [])
            self.episodes = payload.get("episodes", [])
        except Exception:
            pass

    def _save_numeric(self) -> None:
        self._num_path().parent.mkdir(parents=True, exist_ok=True)
        self._num_path().write_text(json.dumps({
            "structure": self.structure, "edges": self.edges,
            "action_stats": self.action_stats, "directives": self.directives,
            "episodes": self.episodes[-40:],
        }, indent=0))

    # ---------------------------------------------------------------- retain
    def retain(self, content: str, *, type: str, entities: list[str] | None = None,
               tags: list[str] | None = None, meta: dict | None = None) -> dict:
        metadata = {"type": type, "entities": ",".join(entities or []),
                    **{k: str(v) for k, v in (meta or {}).items()}}
        result = self.local.retain(self.bank_id, content, metadata=metadata,
                                   tags=tags or [])
        if self.cloud:
            try:
                self.cloud.retain(self.bank_id, content, metadata=metadata,
                                  tags=tags or [])
            except Exception as exc:
                self._log("retain", f"cloud retain failed: {exc}", ok=False)
        self._log("retain", f"{type} · {len(content)} chars · {result['id']}")
        return result

    def retain_episode(self, *, incident: dict, estate: Estate, sim: dict,
                       plan: dict, verdict: str, analyst: str,
                       recalled_ids: list[str]) -> dict:
        """Write the episode, its facts, its directives and its statistics."""
        estate_id = estate.id
        top = sim["paths"][:2]
        lines = [
            f"{incident['id']} · {estate.name} · patient zero "
            f"{sim['patientZero']} · R0 {sim['kpis']['r0']} · "
            f"reach {round(sim['kpis']['reachProbability'] * 100)}% · "
            f"P90 loss ${sim['kpis']['lossP90']:,} · plan "
            f"{', '.join(a['kind'] + ' ' + a['node_label'] for a in plan['actions'])} · "
            f"coverage {round(plan['coverage'] * 100)}% at "
            f"{round(plan['disruption'] * 100)}% disruption · verdict {verdict}"
        ]
        self.retain("\n".join(lines), type="episode",
                    entities=[estate_id, sim["patientZeroId"]]
                    + [n for p in top for n in p["nodes"][:4]],
                    tags=[f"estate:{estate_id}", f"verdict:{verdict}",
                          f"p0:{sim['patientZeroId']}"],
                    meta={"incident": incident["id"], "coverage": plan["coverage"],
                          "verdict": verdict})

        facts = []
        for path in top:
            facts.append(
                f"In {estate.name}, the chain {' → '.join(path['labels'])} "
                f"carried {round(path['prob'] * 100)}% of outbreaks and "
                f"${path['conditionalLoss']:,} conditional loss in "
                f"{path['minutes']} minutes.")
        facts.append(
            f"Rotating patient zero alone covers "
            f"{round(plan['baseline']['coverage'] * 100)}% of residual paths; "
            f"the chosen portfolio covers {round(plan['coverage'] * 100)}% "
            f"for {round(plan['disruption'] * 100)}% disruption.")
        for fact in facts[:4]:
            self.retain(fact, type="fact",
                        entities=[estate_id] + sim["patientZeroId"].split("-"),
                        tags=[f"estate:{estate_id}"],
                        meta={"incident": incident["id"]})

        # procedural memory: which actions bought coverage per unit disruption
        for action in plan["actions"]:
            stat = self.action_stats.setdefault(action["kind"], {
                "picks": 0, "coverage": 0.0, "disruption": 0.0})
            stat["picks"] += 1
            stat["coverage"] += action.get("coverage", 0.0)
            stat["disruption"] += action["disruption"]

        # episodic memory: seed action sets that worked
        self.episodes.append({
            "incident": incident["id"],
            "estate": estate_id, "p0": sim["patientZeroId"],
            "actions": [a["kind"] for a in plan["actions"]],
            "targets": [a["target"] for a in plan["actions"]],
            "coverage": plan["coverage"], "disruption": plan["disruption"],
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.episodes = self.episodes[-40:]

        # structural posteriors from the observed edge fire rates
        for item in sim["edgeFire"]:
            fire = item["fire"]
            if fire <= 0.01:
                continue
            rec = self.edges.setdefault(item["id"], {"alive": 0.0, "dead": 0.0})
            rec["alive"] += fire * 12.0
            rec["dead"] += (1 - fire) * 12.0
            edge = next((e for e in estate.edges if e.key == item["id"]), None)
            if edge:
                s = self.structure.setdefault(edge.rel, {"alive": 1.0, "dead": 1.0})
                s["alive"] += fire * 6.0
                s["dead"] += (1 - fire) * 6.0

        # analyst directive: what this verdict should change next time
        preferred = [a["kind"] for a in plan["actions"]
                     if a["kind"] in ("revoke_sessions", "remove_oauth", "step_up_mfa")]
        if preferred:
            directive = {
                "kind": "prefer_action", "target": preferred[0],
                "text": f"Analyst {analyst}: prefer {preferred[0].replace('_', ' ')} "
                        f"before rotation in {estate.name}.",
                "value": 0.6, "incident": incident["id"],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            self.directives.append(directive)
            self.retain(directive["text"], type="directive",
                        entities=[estate_id], tags=["directive", f"estate:{estate_id}"],
                        meta={"kind": "prefer_action", "target": preferred[0]})
        if verdict == "incomplete":
            worst = plan["actions"][-1] if plan["actions"] else None
            if worst:
                directive = {
                    "kind": "avoid_rotation", "target": worst["target"],
                    "text": f"Hold automated rotation of {worst['node_label']} until "
                            f"consumers are re-inventoried (verdict {incident['id']}).",
                    "incident": incident["id"],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                self.directives.append(directive)
                self.retain(directive["text"], type="directive",
                            entities=[estate_id, worst["target"]], tags=["directive"],
                            meta={"kind": "avoid_rotation", "target": worst["target"]})

        # utility feedback for memories that were actually recalled
        if recalled_ids:
            self.local.feedback(recalled_ids, good=(verdict != "incomplete"))
        self._save_numeric()
        added_directives = (1 if preferred else 0) + (1 if verdict == "incomplete" and plan["actions"] else 0)
        self._log("learn", f"episode {incident['id']} retained · "
                           f"{len(facts)} facts · {added_directives} directive(s)")
        return {"facts": len(facts), "directives": added_directives,
                "edgesUpdated": len(sim["edgeFire"])}

    # ---------------------------------------------------------------- recall
    def recall(self, query: str, max_results: int = 6) -> dict:
        t0 = time.perf_counter()
        result = self.local.recall(self.bank_id, query, max_results=max_results)
        if self.cloud:
            try:
                cloud = self.cloud.recall(self.bank_id, query, max_results=4)
                if cloud.get("results"):
                    result["cloud"] = cloud["results"]
                    result["backend"] = "cloud+local"
            except Exception as exc:
                self._log("recall", f"cloud recall failed: {exc}", ok=False)
        result["latencyMs"] = round((time.perf_counter() - t0) * 1000, 1)
        result["mode"] = self.mode
        self._log("recall", f"{len(result.get('results', []))} memories · "
                            f"{result['latencyMs']}ms")
        return result

    def beliefs(self, estate: Estate, gnn_node_prior: dict | None = None) -> Beliefs:
        """Turn memory into the numeric object the simulator understands."""
        relation_mult: dict[str, float] = {}
        for rel, stat in self.structure.items():
            base = {"contains_secret": 0.74, "mounts": 0.66, "can_assume": 0.55,
                    "can_read": 0.71, "can_write": 0.62, "can_deploy": 0.48,
                    "can_invoke": 0.81, "push_access": 0.59, "member_of": 0.40,
                    "owns": 0.12, "trusts": 0.44, "authenticates": 0.33}.get(rel, 0.5)
            n = stat["alive"] + stat["dead"]
            mean = stat["alive"] / max(1e-9, n)
            # Shrink toward the prior: one outbreak is evidence, not a verdict.
            ratio = 1.0 + (mean / base - 1.0) * (n / (n + 24.0))
            relation_mult[rel] = round(float(np.clip(ratio, 0.86, 1.18)), 3)
        edge_mult: dict[str, float] = {}
        for key, stat in self.edges.items():
            n = stat["alive"] + stat["dead"]
            mean = stat["alive"] / max(1e-9, n)
            edge = next((e for e in estate.edges if e.key == key), None)
            if edge:
                ratio = 1.0 + (mean / max(0.02, edge.p) - 1.0) * (n / (n + 30.0))
                edge_mult[key] = round(float(np.clip(ratio, 0.82, 1.22)), 3)
        # action priors: coverage bought per unit of disruption, normalised
        priors: dict[str, float] = {}
        for kind, stat in self.action_stats.items():
            if stat["picks"]:
                ratio = (stat["coverage"] / max(1e-6, stat["disruption"])) / stat["picks"]
                priors[kind] = round(float(np.clip(ratio / 8.0, 0.0, 1.0)), 3)
        directives = self.directives[-6:]
        # case-based memory: portfolios that worked on this estate, best first
        portfolios = sorted(
            [e for e in self.episodes if e.get("estate") == estate.id],
            key=lambda e: -e.get("coverage", 0.0))[:10]
        focus_targets: list[str] = []
        for episode in portfolios[:5]:
            for target in episode.get("targets", []):
                if target not in focus_targets:
                    focus_targets.append(target)
        notes = []
        if relation_mult:
            notes.append(f"{len(relation_mult)} relation priors learned")
        if edge_mult:
            notes.append(f"{len(edge_mult)} edge posteriors")
        if directives:
            notes.append(f"{len(directives)} analyst directives")
        if portfolios:
            notes.append(f"{len(portfolios)} recalled portfolios")
        return Beliefs(
            edge_mult=edge_mult, relation_mult=relation_mult,
            node_prior={k: v for k, v in (gnn_node_prior or {}).items()},
            directives=directives, action_priors=priors,
            portfolios=portfolios, focus_targets=focus_targets,
            notes=notes, source="hindsight" if (self.edges or self.directives or portfolios) else "cold",
            recalled=len(self.local.records),
        )

    def stats(self) -> dict:
        s = self.local.stats()
        s.update({
            "mode": self.mode,
            "edgesLearned": len(self.edges),
            "relationsLearned": len(self.structure),
            "directives": len(self.directives),
            "episodes": len(self.episodes),
            "actionsLearned": len(self.action_stats),
            "recent": self._op_log[-12:],
        })
        return s

    def _log(self, kind: str, text: str, ok: bool = True) -> None:
        self._op_log.append({
            "kind": kind, "text": text, "ok": ok,
            "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
        self._op_log = self._op_log[-60:]

    def reset(self) -> None:
        self.local.records = []
        self.local._rebuild()
        self.local._save()
        self.structure, self.edges, self.action_stats = {}, {}, {}
        self.directives, self.episodes = [], []
        self._save_numeric()
        self._op_log = []
