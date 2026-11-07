"""Simulated response choreography with audit, health checks and rollback.

Nothing here touches a real provider.  The point is to prove the product
understands operational sequencing: resolve consumers, issue the replacement,
roll it out in waves, verify, only then revoke, and keep rollback possible
until the final step.  Every step is idempotent and carries an audit id.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from .estate import Estate

WAVES = [("canary", 5), ("10%", 18), ("100%", 34)]


def _rid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


def choreograph(estate: Estate, actions: list[dict], *, actor: str,
                clock: datetime | None = None) -> dict:
    clock = clock or datetime.now(timezone.utc)
    steps: list[dict] = []
    audit: list[dict] = []
    seq = 0

    def log(kind: str, text: str, *, step: str = "") -> None:
        nonlocal seq
        seq += 1
        audit.append({
            "id": f"AU-{seq:04d}",
            "at": clock.isoformat().replace("+00:00", "Z"),
            "actor": actor,
            "kind": kind,
            "step": step,
            "text": text,
        })

    log("approval", f"Plan approved by {actor} · {len(actions)} interventions")
    total_minutes = 0

    for action in actions:
        node = estate.nodes[action["target"]]
        consumers = [
            estate.nodes[e.dst].label for e in estate.out.get(action["target"], [])
            if estate.nodes[e.dst].kind in ("workload", "pipeline", "repo")
        ][:4]
        step_id = action["id"]
        idem = _rid(action["kind"], action["target"], str(clock))

        steps.append({
            "id": step_id, "phase": "resolve", "action": action["label"],
            "target": node.label, "status": "ok", "minutes": 3,
            "detail": f"Resolved {len(consumers)} consumers"
                      + (f": {', '.join(consumers)}" if consumers else " (no live consumers)"),
        })
        log("resolve", f"{node.label}: {len(consumers)} consumers resolved", step=step_id)
        total_minutes += 3

        if action["kind"] in ("rotate", "disable"):
            new_ref = f"v{(int(clock.timestamp()) % 9000) + 1000}-{idem[:4]}"
            steps.append({
                "id": step_id, "phase": "issue", "action": "Issue replacement",
                "target": node.label, "status": "ok", "minutes": 2,
                "detail": f"New credential {new_ref} minted with equal-or-narrower scope · "
                          f"old credential still valid (dual-key overlap)",
            })
            log("issue", f"{node.label}: replacement {new_ref} issued; old key retained",
                step=step_id)
            total_minutes += 2
            for wave, pct in WAVES:
                steps.append({
                    "id": step_id, "phase": f"deploy:{wave}",
                    "action": f"Roll out · {wave}", "target": node.label,
                    "status": "ok", "minutes": 4,
                    "detail": f"{pct}% of consumers on the new key · auth success 100%",
                })
                total_minutes += 4
            steps.append({
                "id": step_id, "phase": "verify", "action": "Health verification",
                "target": node.label, "status": "ok", "minutes": 3,
                "detail": "Auth success 99.98% · p95 latency +2ms · 0 consumer failures",
            })
            log("verify", f"{node.label}: health window passed; safe to revoke old key",
                step=step_id)
            steps.append({
                "id": step_id, "phase": "revoke", "action": "Revoke old credential",
                "target": node.label, "status": "ok", "minutes": 1,
                "detail": "Old fingerprint disabled · audit record sealed · rollback closed",
            })
            log("revoke", f"{node.label}: old credential revoked (idempotency {idem})",
                step=step_id)
            total_minutes += 4
        elif action["kind"] in ("revoke_sessions", "remove_oauth"):
            steps.append({
                "id": step_id, "phase": "apply", "action": action["label"],
                "target": node.label, "status": "ok", "minutes": 2,
                "detail": "Sessions/grant invalidated · new authentication required · reversible",
            })
            log("apply", f"{node.label}: {action['label']} applied", step=step_id)
            total_minutes += 2
        elif action["kind"] == "deny_asn":
            steps.append({
                "id": step_id, "phase": "apply", "action": action["label"],
                "target": node.label, "status": "ok", "minutes": 2,
                "detail": "Egress/ingress rule deployed at the edge · external use blocked",
            })
            log("apply", f"ASN deny rule installed for {node.label}", step=step_id)
            total_minutes += 2
        elif action["kind"] == "step_up_mfa":
            steps.append({
                "id": step_id, "phase": "apply", "action": action["label"],
                "target": node.label, "status": "ok", "minutes": 5,
                "detail": "Conditional-access policy enforced for the credential owner",
            })
            log("apply", f"{node.label}: step-up MFA enforced", step=step_id)
            total_minutes += 5
        elif action["kind"] == "reduce_scope":
            steps.append({
                "id": step_id, "phase": "apply", "action": action["label"],
                "target": node.label, "status": "ok", "minutes": 12,
                "detail": "Data-plane scopes removed · production reads unaffected",
            })
            log("apply", f"{node.label}: scope reduced", step=step_id)
            total_minutes += 12

    log("complete", f"Containment complete · {len(steps)} steps · "
                    f"{total_minutes} simulated minutes · audit sealed")
    return {
        "ok": True,
        "steps": steps,
        "audit": audit,
        "simulatedMinutes": total_minutes,
        "idempotency": [_rid(a["kind"], a["target"], str(clock)) for a in actions],
        "health": {"authSuccess": 0.9998, "latencyDeltaMs": 2,
                   "failedChecks": 0, "rolledBack": False},
        "posture": "contained",
    }
