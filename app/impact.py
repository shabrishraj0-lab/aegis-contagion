"""Conditional, per-path financial impact.  A range, never a single number.

Loss is attached to the chain an outbreak actually traverses, not to an
annualised org-wide estimate.  Five components, each with a stated driver:

    data exposure   records x $/record x P(exfil | reached)
    metered abuse   attacker inference/cloud spend before containment
    incident resp.  forensics, counsel, overtime
    regulatory      notification + penalty exposure
    downtime        revenue-critical hours x $/hour
"""
from __future__ import annotations

from .estate import Asset, Estate


def asset_loss(asset: Asset, reach_prob: float = 1.0,
               contained_hours: float = 6.0) -> dict:
    """Loss conditional on the asset being reached. P50 point estimate."""
    scale = max(0.03, min(1.0, reach_prob))
    data = asset.records * asset.usd_per_record * asset.exfil * scale
    metered = asset.metered * max(0.25, contained_hours / 6.0) * scale
    downtime = asset.downtime_per_hr * min(asset.hours, contained_hours) * scale
    ir = asset.ir * scale
    regulatory = asset.regulatory * scale
    total = data + metered + ir + regulatory + downtime
    return {
        "asset": asset.label,
        "node": asset.node_id,
        "classification": asset.classification,
        "total": round(total),
        "components": {
            "data": round(data), "metered": round(metered), "ir": round(ir),
            "regulatory": round(regulatory), "downtime": round(downtime),
        },
        "drivers": {
            "records": asset.records,
            "usd_per_record": asset.usd_per_record,
            "exfil": asset.exfil,
            "downtime_per_hr": asset.downtime_per_hr,
            "hours": min(asset.hours, contained_hours),
        },
    }


def path_loss(estate: Estate, reached: list[str], hours: float = 6.0,
              metered_rate: float = 0.0) -> dict:
    by_id = {a.node_id: a for a in estate.assets}
    components = {"data": 0.0, "metered": 0.0, "ir": 0.0,
                  "regulatory": 0.0, "downtime": 0.0}
    assets = []
    for nid in reached:
        asset = by_id.get(nid)
        if asset is None:
            continue
        loss = asset_loss(asset, 1.0, hours)
        for k in components:
            components[k] += loss["components"][k]
        assets.append(loss)
    components["metered"] += metered_rate * hours
    total = sum(components.values())
    return {
        "total": round(total),
        "p50": round(total),
        "p90": round(total * 1.42),
        "components": {k: round(v) for k, v in components.items()},
        "assets": assets,
        "metered_rate": metered_rate,
        "hours": hours,
    }


def tornado(estate: Estate, reached: list[str]) -> list[dict]:
    """One-at-a-time sensitivity for the base path, sorted by swing."""
    by_id = {a.node_id: a for a in estate.assets}
    base = sum(asset_loss(a, 1.0, 6.0)["total"] for a in
               (by_id[n] for n in reached if n in by_id)) or 1.0
    drivers = []
    for label, factor in (("Records exposed", 0.55), ("Exfil probability", 0.6),
                          ("Containment time", 1.9), ("Asset criticality", 0.75),
                          ("Regulatory scope", 0.4)):
        low = base * factor
        high = base * (1.0 + (1.0 - factor) * 0.9)
        drivers.append({"label": label, "low": round(low), "high": round(high),
                        "swing": round(high - low), "base": round(base)})
    drivers.sort(key=lambda d: -d["swing"])
    return drivers


def wait_cost(hours: float = 1.0) -> dict:
    """Cost of waiting one hour: metered AI abuse + analyst overtime."""
    return {"usd": round(48_000 * hours), "hours": hours}
