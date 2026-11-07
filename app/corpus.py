"""Synthetic incident-memo corpus for CONTAGION-LM.

The corpus is deliberately narrow and repetitive.  A small model trained from
scratch can learn the register and the argument order of a security memo; it
cannot invent facts.  That is exactly the contract: the model drafts, the
grounding validator decides whether a draft survives.
"""
from __future__ import annotations

import random

ESTATES = ["meridian health", "northwind payments", "orbital logistics",
           "atlas retail", "harbor energy", "northstar media"]
P0 = ["openai content prod", "github actions release", "okta provisioner",
      "aws deploy", "stripe live payouts", "vault unseal", "azure finance etl"]
ACTIONS = ["rotate the ci secret", "revoke the oauth grant",
           "remove the release token", "enforce step up mfa",
           "reduce the role scope", "deny the source asn"]
ASSETS = ["production customer archive", "the warehouse", "the finance ledger",
          "model artifacts", "the payout api"]
VERDICTS = ["contained", "incomplete", "contained after rollback",
            "escalated to the owner"]
OPENERS = [
    "outbreak memo .",
    "incident note .",
    "containment record .",
    "credential outbreak summary .",
    "soc memo .",
]
CLOSERS = [
    "the path is the risk .",
    "rotate the cut vertex , not the alarm .",
    "detection is not containment .",
    "price the path , then cut it .",
    "memory changed the plan .",
]


def _memo(rng: random.Random) -> str:
    estate = rng.choice(ESTATES)
    p0 = rng.choice(P0)
    asset = rng.choice(ASSETS)
    actions = rng.sample(ACTIONS, rng.choice([1, 2, 2, 3]))
    r0 = round(rng.uniform(1.2, 9.8), 1)
    reach = rng.choice([18, 24, 31, 38, 44, 52, 61, 67])
    ttc = rng.choice([12, 19, 24, 28, 33, 38, 44, 51, 63])
    loss = rng.choice([0.4, 0.9, 1.2, 1.8, 2.3, 2.9, 3.4, 4.1, 5.6, 7.2])
    cov = rng.choice([41, 48, 55, 62, 68, 73, 78, 81, 87, 91, 94])
    disc = rng.choice([4, 6, 7, 8, 9, 11, 12, 14])
    verdict = rng.choice(VERDICTS)
    opener = rng.choice(OPENERS)
    closer = rng.choice(CLOSERS)
    parts = [
        opener,
        f"estate {estate} .",
        f"patient zero {p0} .",
        f"r zero {r0} .",
        f"reachability {reach} percent .",
        f"median time to crown jewel {ttc} minutes .",
        f"conditional loss {loss} million .",
        f"portfolio {' and '.join(actions)} .",
        f"cut {cov} percent of paths for {disc} percent disruption .",
        f"primary asset {asset} .",
        f"verdict {verdict} .",
    ]
    if rng.random() < 0.55:
        parts.append(f"dominant chain crossed the release pipeline .")
    if rng.random() < 0.40:
        parts.append(f"recalled a prior outbreak and moved earlier .")
    parts.append(closer)
    return " ".join(parts)


def build_corpus(n: int = 1400, seed: int = 77) -> list[str]:
    rng = random.Random(seed)
    docs = [_memo(rng) for _ in range(n)]
    # a few hand-written anchors so the model sees canonical phrasing
    docs += [
        "outbreak memo . estate meridian health . patient zero openai content prod ."
        " r zero 7.4 . reachability 64 percent . median time to crown jewel 38 minutes ."
        " conditional loss 2.3 million . portfolio rotate the ci secret and revoke the"
        " oauth grant . cut 81 percent of paths for 12 percent disruption ."
        " primary asset production customer archive . verdict contained ."
        " the path is the risk .",
        "the scanner found a key . the agent found the outbreak . it priced the path"
        " and cut the smallest set of credentials that breaks it .",
        "detection is not containment . rotate the cut vertex , not the alarm .",
    ]
    return docs
