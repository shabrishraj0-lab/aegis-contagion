"""AEGIS Contagion — HTTP surface.

A small FastAPI app: everything is local, no provider call is required for the
demo.  The frontend is plain HTML/CSS/JS served from ../web.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import SCENARIOS, AegisAgent
from .estate import ESTATE_SPECS, get_estate, patient_zero_options

WEB = Path(__file__).resolve().parent.parent / "web"
DATA = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(title="AEGIS Contagion", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

_agent: AegisAgent | None = None
_lock = threading.Lock()


def agent() -> AegisAgent:
    global _agent
    with _lock:
        if _agent is None:
            _agent = AegisAgent()
        return _agent


class RunBody(BaseModel):
    estate: str = "meridian"
    patientZero: str = "key-openai-content"
    trials: int = 2600
    horizon: int = 90
    policy: str = "balanced"
    useMemory: bool = True
    budget: float = 0.08


class ExecuteBody(BaseModel):
    incidentId: str
    actor: str = "soc-analyst"
    verdict: str = "contained"


class CampaignBody(BaseModel):
    episodes: int = 6
    trials: int = 900
    budget: float = 0.08


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "ts": time.time()}


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    return agent().bootstrap()


@app.get("/api/graph")
def graph(estate: str = Query("meridian")) -> dict:
    if estate not in {s["id"] for s in ESTATE_SPECS}:
        raise HTTPException(404, "unknown estate")
    return get_estate(estate).to_dict()


@app.get("/api/patient-zeros")
def patient_zeros(estate: str = "meridian") -> dict:
    return {"options": patient_zero_options(get_estate(estate))}


@app.post("/api/run")
def run(body: RunBody) -> dict:
    t0 = time.perf_counter()
    try:
        incident = agent().run(
            body.estate, body.patientZero, trials=max(300, min(6000, body.trials)),
            horizon=max(30, min(240, body.horizon)), policy=body.policy,
            use_memory=body.useMemory,
            disruption_budget=max(0.03, min(0.25, body.budget)))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    incident["requestMs"] = round((time.perf_counter() - t0) * 1000)
    return incident


@app.post("/api/execute")
def execute(body: ExecuteBody) -> dict:
    try:
        return agent().execute(body.incidentId, actor=body.actor,
                               verdict=body.verdict)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/campaign")
def campaign(body: CampaignBody) -> dict:
    return agent().run_campaign(episodes=max(1, min(6, body.episodes)),
                                trials=max(300, min(2000, body.trials)),
                                disruption_budget=max(0.03, min(0.25, body.budget)))


@app.get("/api/incident/{incident_id}")
def incident(incident_id: str) -> dict:
    inc = agent().incidents.get(incident_id)
    if inc is None:
        raise HTTPException(404, "unknown incident")
    return inc


@app.get("/api/memory")
def memory(q: str = Query("cut vertex"), limit: int = 8) -> dict:
    return agent().memory.recall(q, max_results=limit)


@app.get("/api/memory/stats")
def memory_stats() -> dict:
    mem = agent().memory
    return {"stats": mem.stats(),
            "records": mem.local.list_memories("aegis-contagion", 60),
            "directives": mem.directives[-10:],
            "episodes": mem.episodes[-10:],
            "actionStats": mem.action_stats,}


@app.post("/api/memory/reset")
def memory_reset() -> dict:
    agent().memory.reset()
    return {"ok": True, "stats": agent().memory.stats()}


@app.get("/api/eval")
def evaluation() -> dict:
    a = agent()
    camp = None
    if (DATA / "campaign.json").exists():
        camp = json.loads((DATA / "campaign.json").read_text())
    return {
        "gnn": a.gnn_meta,
        "lm": {k: v for k, v in a.lm_meta.items() if k != "tokenizer"},
        "memory": a.memory.stats(),
        "campaign": camp,
        "modelCard": {
            "gnn": {
                "name": "ContagionGNN",
                "task": "node compromise probability + edge transmission probability",
                "features": "21 node features · 9 edge features · 6 relation groups",
                "training": "weak supervision from 260 outbreak simulations per estate",
                "honesty": "topology-level holdout; labels are simulator-generated",
            },
            "lm": {
                "name": "CONTAGION-LM",
                "task": "draft prose for an incident memo, never source a number",
                "training": "1,600 synthetic memos, trained from scratch in NumPy",
                "honesty": "small domain model; grounded synthesiser has final say",
            },
        },
    }


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": SCENARIOS}


@app.get("/api/design")
def design() -> dict:
    return agent().bootstrap()["system"]


@app.exception_handler(Exception)
def unhandled(_request, exc: Exception):
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)
