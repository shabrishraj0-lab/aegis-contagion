"""Synthetic — but structurally honest — machine-identity estate.

The graph is heterogeneous on purpose: credentials do not connect to other
credentials, they connect through the resources that hold, delegate or honour
them.  Every edge is evidence, not fact, so it carries an exploit probability,
a time cost, a confidence and a last-verified timestamp.  Three seeded estates
share the same generative motifs so a policy learned on one transfers to the
others (that transfer is what the memory layer is measured on).

Node kinds
    credential  workload  human  repo  pipeline  role  datastore  ai_endpoint  idp
Edge relations
    contains_secret  mounts  can_assume  can_read  can_write  can_deploy
    can_invoke  push_access  member_of  owns  trusts  authenticates
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

KINDS = ("credential", "workload", "human", "repo", "pipeline", "role",
         "datastore", "ai_endpoint", "idp")
RELATIONS = ("contains_secret", "mounts", "can_assume", "can_read", "can_write",
             "can_deploy", "can_invoke", "push_access", "member_of", "owns",
             "trusts", "authenticates")

# relation -> (base exploit probability, base time cost in minutes, confidence)
REL_BASE = {
    "contains_secret": (0.74, 4.0, 0.96),
    "mounts":          (0.66, 2.0, 0.93),
    "can_assume":      (0.55, 11.0, 0.90),
    "can_read":        (0.71, 7.0, 0.95),
    "can_write":       (0.62, 9.0, 0.92),
    "can_deploy":      (0.48, 16.0, 0.88),
    "can_invoke":      (0.81, 3.0, 0.97),
    "push_access":     (0.59, 6.0, 0.91),
    "member_of":       (0.40, 2.0, 0.86),
    "owns":            (0.12, 1.0, 0.83),
    "trusts":          (0.44, 13.0, 0.79),
    "authenticates":   (0.33, 3.0, 0.84),
}

SERVICE_CATALOG = [
    ("aws", "Cloud"), ("gcp", "Cloud"), ("azure", "Cloud"),
    ("github", "Code host"), ("gitlab", "Code host"),
    ("openai", "AI platform"), ("anthropic", "AI platform"),
    ("snowflake", "Data warehouse"), ("vault", "Secrets"),
    ("okta", "Identity"), ("stripe", "Payments"), ("npm", "Package registry"),
]

@dataclass
class Node:
    id: str
    kind: str
    label: str
    service: str = ""
    critical: bool = False
    value: float = 0.0
    records: int = 0
    detail: str = ""
    meta: dict = field(default_factory=dict)
    hops_from_p0: int = 0

@dataclass
class Edge:
    src: str
    dst: str
    rel: str
    p: float
    t: float
    conf: float
    source: str = "api"
    verified_min: int = 12
    key: str = ""
    def __post_init__(self) -> None:
        if not self.key:
            self.key = f"{self.src}|{self.rel}|{self.dst}"

@dataclass
class Asset:
    node_id: str
    label: str
    classification: str
    records: int
    usd_per_record: float
    exfil: float
    ir: float
    regulatory: float
    downtime_per_hr: float
    hours: float
    metered: float = 0.0
    tier: str = "confidential"
    value: float = 0.0


ESTATE_SPECS = [
    {"id": "meridian", "name": "Meridian Health", "seed": 20261107,
     "records": 2_100_000, "region": "US-EAST", "profile": "hospital"},
    {"id": "northwind", "name": "Northwind Payments", "seed": 77113,
     "records": 940_000, "region": "EU-WEST", "profile": "fintech"},
    {"id": "orbital", "name": "Orbital Logistics", "seed": 90317,
     "records": 380_000, "region": "AP-SOUTH", "profile": "logistics"},
]


class Estate:
    def __init__(self, spec: dict) -> None:
        self.id = spec["id"]
        self.name = spec["name"]
        self.seed = spec["seed"]
        self.records = spec["records"]
        self.region = spec["region"]
        self.profile = spec["profile"]
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._rng = random.Random(self.seed)
        self.assets: list[Asset] = []
        self._build()
        self._index()
        self._layout()

    # ------------------------------------------------------------------ build
    def _node(self, nid: str, kind: str, label: str, **kw) -> Node:
        node = Node(id=nid, kind=kind, label=label, **kw)
        self.nodes[nid] = node
        return node

    def _edge(self, src: str, dst: str, rel: str, *, mult: float = 1.0,
              conf: float | None = None, source: str = "api") -> Edge:
        base_p, base_t, base_c = REL_BASE[rel]
        rng = self._rng
        p = min(0.97, max(0.03, base_p * mult * rng.uniform(0.88, 1.12)))
        t = max(1.0, base_t * rng.uniform(0.75, 1.35))
        edge = Edge(src, dst, rel, round(p, 4), round(t, 1),
                    round(conf if conf is not None else base_c * rng.uniform(0.94, 1.0), 3),
                    source, rng.randint(2, 44))
        self.edges.append(edge)
        return edge

    def _build(self) -> None:
        r = self._rng
        prof = self.profile

        # ---- humans & teams ------------------------------------------------
        humans = [
            ("u-rita", "Rita Okonkwo", "Platform", "staff"),
            ("u-luca", "Luca Ferrari", "Data", "senior"),
            ("u-mei", "Mei Tanaka", "AI Engineering", "staff"),
            ("u-jide", "Jide Okafor", "AI Engineering", "senior"),
            ("u-sanjay", "Sanjay Iyer", "Security", "staff"),
            ("u-hannah", "Hannah Weber", "Finance Ops", "manager"),
            ("u-vikram", "Vikram Sharma", "AI Engineering", "senior"),
            ("u-jake", "Jake Miller", "Contractor", "contractor"),
            ("u-priya", "Priya Singh", "Developer Experience", "staff"),
            ("u-ivan", "Ivan Volkov", "Identity", "staff"),
            ("u-simran", "Simran Kaur", "Payments", "manager"),
            ("u-yuki", "Yuki Chen", "Developer Experience", "senior"),
        ]
        for hid, name, team, seniority in humans:
            self._node(hid, "human", name, detail=f"{team} · {seniority}",
                       meta={"team": team, "seniority": seniority,
                             "mfa": seniority != "contractor"})

        # ---- identity providers -------------------------------------------
        self._node("idp-okta", "idp", "Okta Workforce", service="okta",
                   detail="SSO · 1,842 principals", meta={"mfa_coverage": 0.91})
        self._node("idp-entra", "idp", "Entra ID", service="azure",
                   detail="B2B guests · 3 tenants", meta={"mfa_coverage": 0.74})

        # ---- credentials ---------------------------------------------------
        creds = [
            ("key-openai-content", "openai-content-prod", "openai", "api_key", "write", 8, False, 198, 118, True),
            ("key-openai-eval", "openai-eval-harness", "openai", "api_key", "write", 5, True, 96, 12, False),
            ("key-anthropic-bot", "anthropic-support-bot", "anthropic", "api_key", "write", 6, True, 142, 35, False),
            ("key-gh-release", "gh-actions-release", "github", "ci_secret", "admin", 29, True, 377, 19, False),
            ("key-gh-mobile", "gh-mobile-app", "github", "oauth_token", "write", 13, True, 266, 41, False),
            ("key-npm-publish", "npm-publisher", "npm", "oauth_token", "admin", 8, True, 347, 26, False),
            ("key-okta-helpdesk", "okta-helpdesk", "okta", "oauth_token", "write", 18, True, 302, 33, False),
            ("key-okta-provision", "okta-provisioner", "okta", "service_account", "admin", 41, True, 623, 15, False),
            ("key-vault-unseal", "vault-prod-unseal", "vault", "service_account", "admin", 33, True, 810, 11, False),
            ("key-stripe-live", "stripe-live-payouts", "stripe", "api_key", "admin", 21, True, 433, 24, False),
            ("key-snowflake-etl", "prod-warehouse-etl", "snowflake", "service_account", "admin", 37, True, 512, 28, False),
            ("key-snowflake-bi", "snowflake-bi-ro", "snowflake", "oauth_token", "read", 19, True, 288, 61, False),
            ("key-aws-deploy", "prod-aws-deploy", "aws", "service_account", "admin", 44, True, 611, 21, False),
            ("key-aws-backup", "aws-backup-writer", "aws", "api_key", "write", 9, True, 702, 96, False),
            ("key-gcp-ml", "gcp-ml-training", "gcp", "service_account", "admin", 31, True, 244, 17, False),
            ("key-azure-etl", "azure-finance-etl", "azure", "service_account", "write", 16, True, 388, 44, False),
            ("key-gitlab-mirror", "gitlab-mirror", "gitlab", "service_account", "write", 10, True, 590, 54, False),
            ("key-docker-release", "docker-release", "aws", "pat", "write", 6, False, 268, 97, True),
            ("key-datadog", "datadog-ingest", "aws", "api_key", "write", 7, True, 517, 39, False),
            ("key-jenkins", "jenkins-deploy", "aws", "ci_secret", "admin", 26, True, 556, 31, False),
        ]
        for cid, label, service, kind, priv, scopes, mfa, age, rotated, exposed in creds:
            self._node(cid, "credential", label, service=service,
                       detail=f"{kind.replace('_', ' ')} · {priv} · {scopes} scopes",
                       meta={"kind": kind, "privilege": priv, "scopes": scopes,
                             "mfa": mfa, "age_days": age, "rotated_days": rotated,
                             "exposed": exposed, "shared": 3 if cid == "key-docker-release" else 0})

        # ---- workloads -----------------------------------------------------
        workloads = [
            ("wl-content-agent", "content-agent", "openai", "AI agent", 4),
            ("wl-eval-runner", "eval-runner", "openai", "batch job", 2),
            ("wl-support-bot", "support-bot", "anthropic", "AI agent", 3),
            ("wl-release", "release-orchestrator", "github", "CI runner", 12),
            ("wl-mobile-cd", "mobile-cd", "github", "CI runner", 6),
            ("wl-package-publish", "package-publisher", "npm", "CI runner", 5),
            ("wl-helpdesk", "helpdesk-portal", "okta", "service", 7),
            ("wl-vault", "vault-controller", "vault", "controller", 9),
            ("wl-payouts", "payout-api", "stripe", "service", 11),
            ("wl-warehouse", "warehouse-loader", "snowflake", "pipeline", 15),
            ("wl-bi", "bi-gateway", "snowflake", "service", 8),
            ("wl-deploy", "deploy-controller", "aws", "controller", 13),
            ("wl-backup", "backup-worker", "aws", "job", 5),
            ("wl-training", "training-cluster", "gcp", "job", 14),
            ("wl-finance", "finance-etl", "azure", "pipeline", 10),
            ("wl-mirror", "mirror-sync", "gitlab", "job", 4),
            ("wl-admin", "admin-console", "okta", "service", 16),
            ("wl-jenkins", "jenkins-master", "aws", "CI runner", 9),
        ]
        for wid, label, service, note, hops in workloads:
            self._node(wid, "workload", label, service=service,
                       detail=note, meta={"zone": "prod" if hops > 7 else "shared"})

        # ---- repos / pipelines / roles ------------------------------------
        repos = [
            ("repo-web", "orbit/web-client", "public"),
            ("repo-api", "orbit/api-gateway", "private"),
            ("repo-ai", "orbit/content-agent", "private"),
            ("repo-infra", "orbit/infra-live", "private"),
            ("repo-mobile", "orbit/mobile-android", "private"),
            ("repo-npm", "orbit/ui-kit", "public"),
            ("repo-mirror", "orbit/gitlab-mirror", "private"),
        ]
        for rid, label, vis in repos:
            self._node(rid, "repo", label, detail=f"git · {vis}",
                       meta={"visibility": vis,
                             "protected": rid in ("repo-api", "repo-infra")})

        pipelines = [
            ("pipe-release", "release-prod", "github", 14),
            ("pipe-mobile", "mobile-release", "github", 7),
            ("pipe-npm", "npm-publish", "npm", 5),
            ("pipe-jenkins", "jenkins-deploy", "aws", 18),
            ("pipe-etl", "finance-etl", "azure", 22),
        ]
        for pid, label, service, mins in pipelines:
            self._node(pid, "pipeline", label, service=service,
                       detail=f"deploy pipeline · ~{mins} min",
                       meta={"duration_min": mins})

        roles = [
            ("role-aws-prod", "AWSProdDeployRole", "aws", "assume-role · 12h", 0.9),
            ("role-aws-data", "AWSDataEngineer", "aws", "assume-role · 4h", 0.7),
            ("role-gcp-run", "GCPMLRunner", "gcp", "service account", 0.8),
            ("role-azure-contrib", "AzureFinanceContributor", "azure", "app registration", 0.75),
            ("role-k8s-admin", "prod-k8s-admin", "aws", "cluster-admin", 0.95),
            ("role-stripe-admin", "StripePayoutAdmin", "stripe", "restricted key", 0.8),
            ("role-okta-super", "OktaSuperAdmin", "okta", "admin role", 0.9),
            ("role-vault-root", "VaultUnseal", "vault", "unseal shard", 0.85),
        ]
        for rid, label, service, note, sens in roles:
            self._node(rid, "role", label, service=service,
                       detail=note, meta={"sensitivity": sens})

        ais = [
            ("ai-openai", "OpenAI Platform", "openai", 4800),
            ("ai-anthropic", "Anthropic API", "anthropic", 3200),
            ("ai-gcp", "Vertex AI", "gcp", 2100),
        ]
        for aid, label, service, rate in ais:
            self._node(aid, "ai_endpoint", label, service=service,
                       detail=f"metered inference · ${rate:,}/hr abuse rate",
                       meta={"abuse_rate": rate})

        # ---- data stores ---------------------------------------------------
        ds = [
            ("ds-customer-s3", "prod-customer-archive", "S3 bucket", "regulated", 4_400_000, 0.35, 310_000, 3_600_000, 95_000, 6, "phi"),
            ("ds-warehouse", "prod-warehouse", "Snowflake", "regulated", 13_200_000, 0.34, 420_000, 2_400_000, 88_000, 5, "confidential"),
            ("ds-finance", "finance-ledger", "Postgres", "restricted", 1_900_000, 0.28, 180_000, 900_000, 41_000, 4, "restricted"),
            ("ds-model-artifacts", "model-artifacts", "GCS bucket", "internal", 620_000, 0.22, 120_000, 0, 12_000, 3, "internal"),
            ("ds-logs", "observability-logs", "ClickHouse", "internal", 180_000, 0.12, 60_000, 0, 6_000, 2, "internal"),
        ]
        ir_by_tier = {"phi": 310_000, "confidential": 260_000,
                      "restricted": 180_000, "internal": 90_000}
        for (did, label, engine, cls, value, exfil, records, regulatory, down, hours, tier) in ds:
            self._node(did, "datastore", label, service=engine,
                       critical=True, value=value, records=records,
                       detail=f"{engine} · {cls} · {records:,} records",
                       meta={"classification": tier, "exfil": exfil, "hours": hours})
            self.assets.append(Asset(
                node_id=did, label=label, classification=tier, records=records,
                usd_per_record=round(value * exfil / max(1, records), 2),
                exfil=exfil, ir=ir_by_tier[tier], regulatory=regulatory,
                downtime_per_hr=down, hours=hours, metered=0.0, tier=tier,
                value=value))

        # ==================================================================
        # Edges: the actual contagion topology
        # ==================================================================
        # ownership / identity
        for hid, _, _, _ in humans:
            self._edge(hid, "idp-okta", "authenticates", mult=0.7)
        for cid, *_ in creds:
            owner = r.choice([h[0] for h in humans])
            self._edge(owner, cid, "owns", mult=0.6, source="hr")
        # credential -> workload mounts
        mounts = {
            "key-openai-content": "wl-content-agent",
            "key-openai-eval": "wl-eval-runner",
            "key-anthropic-bot": "wl-support-bot",
            "key-gh-release": "wl-release",
            "key-gh-mobile": "wl-mobile-cd",
            "key-npm-publish": "wl-package-publish",
            "key-okta-helpdesk": "wl-helpdesk",
            "key-okta-provision": "wl-admin",
            "key-vault-unseal": "wl-vault",
            "key-stripe-live": "wl-payouts",
            "key-snowflake-etl": "wl-warehouse",
            "key-snowflake-bi": "wl-bi",
            "key-aws-deploy": "wl-deploy",
            "key-aws-backup": "wl-backup",
            "key-gcp-ml": "wl-training",
            "key-azure-etl": "wl-finance",
            "key-gitlab-mirror": "wl-mirror",
            "key-docker-release": "wl-package-publish",
            "key-datadog": "wl-deploy",
            "key-jenkins": "wl-jenkins",
        }
        for cid, wid in mounts.items():
            self._edge(wid, cid, "contains_secret", mult=0.95 if "openai" in cid else 0.86)
            # holding the credential lets an attacker act with the workload's
            # identity, so compromise propagates in both directions
            self._edge(cid, wid, "authenticates", mult=0.62)

        # credentials reach the metered endpoints they are entitled to invoke
        for cid, aid in (("key-openai-content", "ai-openai"),
                         ("key-openai-eval", "ai-openai"),
                         ("key-anthropic-bot", "ai-anthropic"),
                         ("key-gcp-ml", "ai-gcp")):
            self._edge(cid, aid, "can_invoke", mult=0.9)

        # the headline chain: a leaked AI key, its agent workload, the CI
        # secret in the same deployment env, the release pipeline, production.
        self._edge("wl-content-agent", "key-gh-release", "contains_secret",
                   mult=1.25, conf=0.9, source="inferred")
        self._edge("wl-content-agent", "repo-ai", "can_write", mult=0.7)
        self._edge("key-openai-content", "repo-ai", "can_read", mult=0.35,
                   source="inferred", conf=0.62)

        # workloads -> provider endpoints
        self._edge("wl-content-agent", "ai-openai", "can_invoke", mult=1.0)
        self._edge("wl-eval-runner", "ai-openai", "can_invoke")
        self._edge("wl-support-bot", "ai-anthropic", "can_invoke")
        self._edge("wl-training", "ai-gcp", "can_invoke")

        # the release spine (patient zero -> production, four hops)
        self._edge("wl-release", "key-gh-release", "contains_secret", mult=1.15)
        self._edge("key-gh-release", "repo-web", "push_access", mult=1.1)
        self._edge("key-gh-release", "repo-infra", "push_access", mult=0.95)
        self._edge("repo-web", "pipe-release", "can_deploy", mult=1.05)
        self._edge("pipe-release", "role-aws-prod", "can_assume", mult=1.2)
        self._edge("pipe-release", "key-npm-publish", "contains_secret", mult=0.72)
        self._edge("repo-npm", "pipe-npm", "can_deploy", mult=0.95)
        self._edge("key-npm-publish", "repo-npm", "push_access", mult=0.85)
        self._edge("role-aws-prod", "ds-customer-s3", "can_read", mult=1.15)
        self._edge("role-aws-prod", "role-k8s-admin", "can_assume", mult=0.62)
        self._edge("role-k8s-admin", "ds-warehouse", "can_read", mult=0.95)
        self._edge("role-aws-prod", "key-vault-unseal", "contains_secret", mult=0.55,
                   source="inferred", conf=0.78)
        self._edge("key-vault-unseal", "role-vault-root", "can_assume", mult=0.9)

        # warehouse / data spine
        self._edge("wl-warehouse", "ds-warehouse", "can_write", mult=1.1)
        self._edge("wl-warehouse", "ds-warehouse", "can_read", mult=1.05)
        self._edge("wl-bi", "ds-warehouse", "can_read", mult=0.95)
        self._edge("key-snowflake-etl", "role-aws-data", "trusts", mult=0.55,
                   source="inferred", conf=0.74)
        self._edge("role-aws-data", "ds-customer-s3", "can_read", mult=0.9)

        # finance spine
        self._edge("wl-finance", "ds-finance", "can_write", mult=1.15)
        self._edge("key-azure-etl", "role-azure-contrib", "can_assume", mult=1.0)
        self._edge("role-azure-contrib", "ds-finance", "can_read", mult=1.1)
        self._edge("role-azure-contrib", "key-stripe-live", "contains_secret", mult=0.6,
                   source="inferred", conf=0.7)
        self._edge("key-stripe-live", "role-stripe-admin", "can_assume", mult=0.9)

        # ML spine
        self._edge("key-gcp-ml", "role-gcp-run", "can_assume", mult=1.0)
        self._edge("role-gcp-run", "ds-model-artifacts", "can_write", mult=1.1)
        self._edge("wl-training", "ds-model-artifacts", "can_read", mult=0.9)

        # identity / admin spine
        self._edge("key-okta-helpdesk", "idp-okta", "authenticates", mult=0.9)
        self._edge("idp-okta", "wl-helpdesk", "trusts", mult=0.65)
        self._edge("wl-helpdesk", "role-okta-super", "can_assume", mult=0.7)
        self._edge("key-okta-provision", "role-okta-super", "can_assume", mult=1.1)
        self._edge("role-okta-super", "idp-okta", "trusts", mult=1.2)
        self._edge("role-okta-super", "ds-logs", "can_read", mult=0.8)
        self._edge("key-aws-deploy", "role-aws-prod", "can_assume", mult=1.05)
        self._edge("key-jenkins", "pipe-jenkins", "can_deploy", mult=1.1)
        self._edge("pipe-jenkins", "role-k8s-admin", "can_assume", mult=1.0)
        self._edge("key-docker-release", "repo-infra", "push_access", mult=0.78)
        self._edge("key-gh-mobile", "repo-mobile", "push_access", mult=0.95)
        self._edge("repo-mobile", "pipe-mobile", "can_deploy", mult=0.95)
        self._edge("key-anthropic-bot", "ai-anthropic", "can_invoke", mult=1.0)
        self._edge("wl-mirror", "repo-mirror", "can_write", mult=0.8)
        self._edge("key-gitlab-mirror", "repo-mirror", "push_access", mult=0.85)
        self._edge("role-aws-prod", "ds-logs", "can_read", mult=0.7)
        self._edge("role-k8s-admin", "wl-deploy", "trusts", mult=0.85)
        self._edge("wl-deploy", "role-k8s-admin", "can_assume", mult=0.9)

        # lateral movement: a few plausible-but-unverified hops
        self._edge("wl-release", "key-jenkins", "contains_secret", mult=0.32,
                   source="inferred", conf=0.66)
        self._edge("wl-deploy", "key-aws-backup", "contains_secret", mult=0.45,
                   source="inferred", conf=0.71)
        self._edge("wl-admin", "key-okta-helpdesk", "contains_secret", mult=0.5,
                   source="inferred", conf=0.69)
        self._edge("wl-payouts", "ds-finance", "can_read", mult=0.6)
        self._edge("wl-backup", "ds-customer-s3", "can_write", mult=0.75)
        self._edge("ds-logs", "ds-warehouse", "can_read", mult=0.25,
                   source="inferred", conf=0.6)

        # a little procedural noise: co-tenancy edges that make the graph
        # non-obvious without inventing fantasy paths.
        cred_ids = [c[0] for c in creds]
        work_ids = [w[0] for w in workloads]
        for wid in r.sample(work_ids, 6):
            for cid in r.sample(cred_ids, 2):
                if not any(e.src == wid and e.dst == cid for e in self.edges):
                    self._edge(wid, cid, "contains_secret", mult=0.18,
                               source="inferred", conf=0.58)

    # ------------------------------------------------------------------ index
    def _index(self) -> None:
        self.out: dict[str, list[Edge]] = {n: [] for n in self.nodes}
        self.inn: dict[str, list[Edge]] = {n: [] for n in self.nodes}
        for e in self.edges:
            self.out[e.src].append(e)
            self.inn[e.dst].append(e)
        self.critical_ids = [n for n, node in self.nodes.items() if node.critical]
        self.credential_ids = [n for n, node in self.nodes.items() if node.kind == "credential"]

    def _layout(self) -> None:
        """Deterministic seeded 2.5D positions: kind rings + local jitter."""
        r = random.Random(self.seed ^ 0x5F3759DF)
        rings = {
            "credential": 0.42, "workload": 0.58, "repo": 0.72, "pipeline": 0.84,
            "role": 0.94, "datastore": 1.12, "ai_endpoint": 1.02, "idp": 0.88,
            "human": 1.22,
        }
        groups: dict[str, list[Node]] = {k: [] for k in KINDS}
        for node in self.nodes.values():
            groups[node.kind].append(node)
        for kind, members in groups.items():
            members.sort(key=lambda n: n.id)
            radius = rings[kind]
            offset = r.uniform(0, math.tau)
            for i, node in enumerate(members):
                angle = offset + math.tau * i / max(1, len(members))
                rad = radius * (0.82 + 0.18 * r.random())
                node.meta["pos"] = [
                    round(math.cos(angle) * rad, 4),
                    round(r.uniform(-0.26, 0.26), 4),
                    round(math.sin(angle) * rad, 4),
                ]
        # place critical assets with a deliberate outer silhouette
        for i, cid in enumerate(self.critical_ids):
            angle = math.tau * i / max(1, len(self.critical_ids)) + 0.4
            self.nodes[cid].meta["pos"] = [
                round(math.cos(angle) * 1.36, 4), 0.1,
                round(math.sin(angle) * 1.36, 4)]

    # ------------------------------------------------------------------ utils
    def edge_density(self) -> float:
        n = len(self.nodes)
        return round(len(self.edges) / (n * (n - 1) / 2), 4)

    def reachable_value(self) -> float:
        return round(sum(a.value for a in self.assets))

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "region": self.region,
            "profile": self.profile, "seed": self.seed,
            "stats": {
                "nodes": len(self.nodes), "edges": len(self.edges),
                "credentials": len(self.credential_ids),
                "criticalAssets": len(self.assets),
                "reachableValue": self.reachable_value(),
                "density": self.edge_density(),
            },
            "nodes": [
                {"id": n.id, "kind": n.kind, "label": n.label, "service": n.service,
                 "critical": n.critical, "value": n.value, "records": n.records,
                 "detail": n.detail,
                 "pos": n.meta.get("pos", [0, 0, 0]),
                 "privilege": n.meta.get("privilege"),
                 "exposed": n.meta.get("exposed", False),
                 "mfa": n.meta.get("mfa"),
                 "team": n.meta.get("team")}
                for n in self.nodes.values()
            ],
            "edges": [
                {"id": e.key, "src": e.src, "dst": e.dst, "rel": e.rel,
                 "p": e.p, "t": e.t, "conf": e.conf, "source": e.source,
                 "verified": e.verified_min}
                for e in self.edges
            ],
        }


_ESTATES: dict[str, Estate] = {}

def get_estate(estate_id: str = "meridian") -> Estate:
    if estate_id not in _ESTATES:
        spec = next((s for s in ESTATE_SPECS if s["id"] == estate_id), ESTATE_SPECS[0])
        _ESTATES[estate_id] = Estate(spec)
    return _ESTATES[estate_id]

def all_estates() -> list[dict]:
    return [get_estate(s["id"]).to_dict() for s in ESTATE_SPECS]

def patient_zero_options(estate: Estate) -> list[dict]:
    out = []
    for cid in estate.credential_ids:
        node = estate.nodes[cid]
        out.append({
            "id": cid, "label": node.label, "service": node.service,
            "privilege": node.meta.get("privilege"),
            "exposed": node.meta.get("exposed", False),
            "detail": node.detail,
            "risk": round(0.3 + 0.5 * bool(node.meta.get("exposed"))
                          + 0.1 * (node.meta.get("privilege") == "admin"), 2),
        })
    out.sort(key=lambda x: -x["risk"])
    return out
