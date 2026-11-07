"""A heterogeneous GraphSAGE, written from scratch in NumPy.

No PyTorch, no PyG — the forward pass and every gradient are implemented here.
Two message-passing layers with relation-specific weight matrices; a node head
that predicts compromise probability within the horizon, and an edge head that
predicts whether an edge transmits.

Training labels come from the simulator (weak supervision, stated as such):
P(node reached within 6h) and P(edge fires).  Reported metrics are computed on
a topology the model never trained on, and calibration is reported alongside
ranking because these scores drive escalation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .estate import Estate, KINDS

REL_GROUPS = {
    "secret": ["contains_secret", "mounts"],
    "identity": ["authenticates", "trusts", "member_of", "owns"],
    "assume": ["can_assume"],
    "data": ["can_read", "can_write"],
    "deploy": ["can_deploy", "push_access"],
    "invoke": ["can_invoke"],
}
GROUP_INDEX = {rel: i for i, rels in enumerate(REL_GROUPS.values()) for rel in rels}
N_GROUPS = len(REL_GROUPS)


def build_tensors(estate: Estate, beliefs=None):
    nodes = list(estate.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    X = np.zeros((N, 21), dtype=np.float32)
    for i, nid in enumerate(nodes):
        node = estate.nodes[nid]
        kind_onehot = [1.0 if node.kind == k else 0.0 for k in KINDS]
        priv = {"read": 0.25, "write": 0.6, "admin": 1.0}.get(
            node.meta.get("privilege", ""), 0.0)
        out_deg = len(estate.out[nid])
        in_deg = len(estate.inn[nid])
        X[i, :len(KINDS)] = kind_onehot
        X[i, len(KINDS):] = [
            priv, min(1.0, node.meta.get("scopes", 0) / 60.0),
            1.0 if node.meta.get("exposed") else 0.0,
            0.0 if node.meta.get("mfa", True) else 1.0,
            min(1.0, node.meta.get("age_days", 0) / 900.0),
            1.0 if node.critical else 0.0,
            min(1.0, np.log1p(node.value) / np.log1p(15_000_000)),
            min(1.0, np.log1p(out_deg) / np.log1p(20)),
            min(1.0, np.log1p(in_deg) / np.log1p(20)),
            1.0 if node.kind == "ai_endpoint" else 0.0,
            1.0 if node.kind == "credential" else 0.0,
            1.0 if node.kind in ("pipeline", "repo") else 0.0,
        ]
    A = np.zeros((N_GROUPS, N, N), dtype=np.float32)
    for e in estate.edges:
        g = GROUP_INDEX.get(e.rel, 0)
        A[g, idx[e.dst], idx[e.src]] += 1.0
    for g in range(N_GROUPS):
        row = A[g].sum(axis=1, keepdims=True)
        A[g] = A[g] / np.maximum(row, 1.0)
    src = np.array([idx[e.src] for e in estate.edges], dtype=np.int32)
    dst = np.array([idx[e.dst] for e in estate.edges], dtype=np.int32)
    EF = np.zeros((len(estate.edges), 10), dtype=np.float32)
    for i, e in enumerate(estate.edges):
        g = GROUP_INDEX.get(e.rel, 0)
        EF[i, g] = 1.0
        EF[i, N_GROUPS:] = [e.p, min(1.0, np.log1p(e.t) / 4.0), e.conf,
                            1.0 if e.source == "inferred" else 0.0]
    return {"X": X, "A": A, "src": src, "dst": dst, "EF": EF,
            "nodes": nodes, "idx": idx}


class GraphSAGE:
    """Two relation-aware GraphSAGE layers + node and edge heads."""

    def __init__(self, n_in: int = 21, n_hid: int = 48, n_groups: int = N_GROUPS,
                 n_ef: int = 10, seed: int = 4) -> None:
        rng = np.random.default_rng(seed)
        def he(shape):
            fan_in = shape[0] if len(shape) > 1 else shape[0]
            return (rng.standard_normal(shape) * np.sqrt(2.0 / max(1, fan_in))
                    ).astype(np.float32)
        self.p = {
            "W1s": he((n_in, n_hid)), "W1r": he((n_groups, n_in, n_hid)),
            "b1": np.zeros(n_hid, dtype=np.float32),
            "W2s": he((n_hid, n_hid)), "W2r": he((n_groups, n_hid, n_hid)),
            "b2": np.zeros(n_hid, dtype=np.float32),
            "wo": he((n_hid, 1)), "bo": np.zeros(1, dtype=np.float32),
            "We1": he((2 * n_hid + n_ef, 32)), "be1": np.zeros(32, dtype=np.float32),
            "we2": he((32, 1)), "be2": np.zeros(1, dtype=np.float32),
        }
        self.m: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v: dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = 0
        self.temp = 1.0

    # ------------------------------------------------------------------ forward
    def forward(self, t, edge_mask=None):
        X, A, src, dst, EF = t["X"], t["A"], t["src"], t["dst"], t["EF"]
        p = self.p
        # relation aggregation: sum_g (A_g X) W_g, no dense einsum needed
        rel1 = np.zeros((X.shape[0], p["W1r"].shape[2]), dtype=np.float32)
        for g in range(A.shape[0]):
            rel1 += (A[g] @ X) @ p["W1r"][g]
        z1 = X @ p["W1s"] + rel1 + p["b1"]
        h1 = np.maximum(z1, 0)
        rel2 = np.zeros((h1.shape[0], p["W2r"].shape[2]), dtype=np.float32)
        for g in range(A.shape[0]):
            rel2 += (A[g] @ h1) @ p["W2r"][g]
        z2 = h1 @ p["W2s"] + rel2 + p["b2"]
        h2 = np.maximum(z2, 0)
        node_logits = (h2 @ p["wo"] + p["bo"]).ravel()
        e_in = np.concatenate([h2[src], h2[dst], EF], axis=1)
        e_h = np.maximum(e_in @ p["We1"] + p["be1"], 0)
        edge_logits = (e_h @ p["we2"] + p["be2"]).ravel()
        return {"z1": z1, "h1": h1, "z2": z2, "h2": h2,
                "node_logits": node_logits, "e_in": e_in, "e_h": e_h,
                "edge_logits": edge_logits}

    def loss_and_grads(self, t, y_node, y_edge, node_w=1.0, edge_w=1.0):
        cache = self.forward(t)
        A, X, src, dst = t["A"], t["X"], t["src"], t["dst"]
        n = len(y_node)
        sn = 1.0 / (1.0 + np.exp(-cache["node_logits"]))
        se = 1.0 / (1.0 + np.exp(-cache["edge_logits"]))
        eps = 1e-7
        nw = np.full(n, node_w, dtype=np.float32) if np.isscalar(node_w) else node_w
        ew = (np.full(len(y_edge), edge_w, dtype=np.float32)
              if np.isscalar(edge_w) else edge_w)
        node_loss = -np.mean(nw * (y_node * np.log(sn + eps) +
                                   (1 - y_node) * np.log(1 - sn + eps)))
        edge_loss = -np.mean(ew * (y_edge * np.log(se + eps) +
                                   (1 - y_edge) * np.log(1 - se + eps)))
        loss = node_loss + edge_loss
        grads = {}
        dn = nw * (sn - y_node) / n
        de = ew * (se - y_edge) / max(1, len(y_edge))
        p = self.p
        grads["wo"] = cache["h2"].T @ dn[:, None]
        grads["bo"] = np.array([dn.sum()], dtype=np.float32)
        dh2 = dn[:, None] @ p["wo"].T
        grads["we2"] = cache["e_h"].T @ de[:, None]
        grads["be2"] = np.array([de.sum()], dtype=np.float32)
        de_h = de[:, None] @ p["we2"].T
        de_h = de_h * (cache["e_h"] > 0)
        grads["We1"] = cache["e_in"].T @ de_h
        grads["be1"] = de_h.sum(axis=0)
        de_in = de_h @ p["We1"].T
        h = cache["h2"].shape[1]
        np.add.at(dh2, src, de_in[:, :h])
        np.add.at(dh2, dst, de_in[:, h:2 * h])
        dz2 = dh2 * (cache["z2"] > 0)
        grads["W2s"] = cache["h1"].T @ dz2
        grads["W2r"] = np.zeros_like(p["W2r"])
        dh1 = dz2 @ p["W2s"].T
        for g in range(A.shape[0]):
            grads["W2r"][g] = (A[g] @ cache["h1"]).T @ dz2
            dh1 += A[g].T @ (dz2 @ p["W2r"][g].T)
        grads["b2"] = dz2.sum(axis=0)
        dz1 = dh1 * (cache["z1"] > 0)
        grads["W1s"] = X.T @ dz1
        grads["W1r"] = np.zeros_like(p["W1r"])
        for g in range(A.shape[0]):
            grads["W1r"][g] = (A[g] @ X).T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return float(loss), grads, {"node": float(node_loss), "edge": float(edge_loss)}

    # ------------------------------------------------------------------ optim
    def adam(self, grads, lr=0.012, clip=4.0, weight_decay=2e-4):
        self.step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-9))
        for k, g in grads.items():
            g = g * scale + weight_decay * self.p[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mhat = self.m[k] / (1 - b1 ** self.step)
            vhat = self.v[k] / (1 - b2 ** self.step)
            self.p[k] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)

    def predict(self, t):
        c = self.forward(t)
        nl = np.clip(c["node_logits"] / self.temp, -30, 30)
        el = np.clip(c["edge_logits"] / self.temp, -30, 30)
        return 1 / (1 + np.exp(-nl)), 1 / (1 + np.exp(-el))

    def state(self):
        return {k: v.astype(np.float32) for k, v in self.p.items()}

    def load_state(self, state):
        for k, v in state.items():
            if k in self.p and v.shape == self.p[k].shape:
                self.p[k] = v.astype(np.float32)


# ---------------------------------------------------------------------------
# training + evaluation
# ---------------------------------------------------------------------------

def _metrics(y, pred):
    order = np.argsort(-pred)
    y_sorted = y[order]
    hits = np.cumsum(y_sorted)
    precision = hits / (np.arange(len(y)) + 1)
    ap = float((precision * y_sorted).sum() / max(1e-9, y_sorted.sum()))
    # ECE, 10 bins — calibration matters because these scores drive escalation
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    calibration = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred >= lo) & (pred < hi)
        if mask.sum() == 0:
            calibration.append({"bin": f"{lo:.1f}", "pred": None,
                                "actual": None, "n": 0})
            continue
        ece += abs(pred[mask].mean() - y[mask].mean()) * mask.sum() / len(pred)
        calibration.append({"bin": f"{lo:.1f}",
                            "pred": round(float(pred[mask].mean()), 3),
                            "actual": round(float(y[mask].mean()), 3),
                            "n": int(mask.sum())})
    brier = float(np.mean((pred - y) ** 2))
    return {"ap": round(ap, 3), "ece": round(float(ece), 3),
            "brier": round(brier, 3), "calibration": calibration,
            "positiveRate": round(float(y.mean()), 3)}


def _fit_temperature(logits, y) -> float:
    best_t, best_loss = 1.0, 1e9
    for temp in np.arange(0.5, 4.01, 0.1):
        p = 1 / (1 + np.exp(-np.clip(logits / temp, -30, 30)))
        loss = -np.mean(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7))
        if loss < best_loss:
            best_t, best_loss = float(temp), float(loss)
    return best_t


def train_and_save(out_path: str | Path, quick: bool = False) -> dict:
    from .estate import ESTATE_SPECS, Estate
    from .simulator import labels as sim_labels

    t0 = time.time()
    specs = ESTATE_SPECS
    held_spec = {"id": "holdout", "name": "Holdout Estuary", "seed": 555001,
                 "records": 1_200_000, "region": "US-WEST", "profile": "hospital"}
    cal_spec = {"id": "calibration", "name": "Calibration Yard", "seed": 611113,
                "records": 700_000, "region": "EU-NORTH", "profile": "fintech"}
    train_data = []
    for spec in specs:
        estate = Estate(spec)
        y_node, y_edge, _ = sim_labels(estate, trials=180 if quick else 300,
                                       horizon=300, seed=spec["seed"] % 97)
        train_data.append((build_tensors(estate),
                           (y_node >= 0.10).astype(np.float32),
                           (y_edge >= 0.05).astype(np.float32)))
    hold_estate = Estate(held_spec)
    hy_node, hy_edge, _ = sim_labels(hold_estate, trials=260, horizon=300, seed=31)
    hy_node = (hy_node >= 0.10).astype(np.float32)
    hy_edge = (hy_edge >= 0.05).astype(np.float32)
    hold = build_tensors(hold_estate)
    cal_estate = Estate(cal_spec)
    cy_node, cy_edge, _ = sim_labels(cal_estate, trials=240, horizon=300, seed=41)
    cy_node = (cy_node >= 0.10).astype(np.float32)
    cal = build_tensors(cal_estate)

    model = GraphSAGE(seed=4)
    rng = np.random.default_rng(9)
    losses = []
    best_state, best_val = None, 1e9
    epochs = 160 if quick else 340
    for epoch in range(epochs):
        order = rng.permutation(len(train_data))
        total = 0.0
        for i in order:
            t, yn, ye = train_data[i]
            node_w = np.where(yn > 0.5, 2.6, 0.75).astype(np.float32)
            edge_w = np.where(ye > 0.5, 2.0, 0.55).astype(np.float32)
            loss, grads, parts = model.loss_and_grads(t, yn, ye,
                                                       node_w=node_w,
                                                       edge_w=edge_w)
            model.adam(grads, lr=0.018 if epoch < epochs * 0.65 else 0.005)
            total += loss
        losses.append(round(total / len(train_data), 5))
        if epoch % 10 == 0 or epoch == epochs - 1:
            vc = model.forward(cal)
            vp = 1 / (1 + np.exp(-np.clip(vc["node_logits"], -30, 30)))
            val_loss = float(-np.mean(cy_node * np.log(vp + 1e-7)
                                      + (1 - cy_node) * np.log(1 - vp + 1e-7)))
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.copy() for k, v in model.p.items()}
    if best_state is not None:
        model.p = best_state
    raw = model.forward(cal)["node_logits"]
    model.temp = _fit_temperature(raw, cy_node)

    hn, he = model.predict(hold)
    node_metrics = _metrics(hy_node, hn)
    edge_metrics = _metrics(hy_edge, he)
    metrics = {
        "node": node_metrics, "edge": edge_metrics,
        "trainSeconds": round(time.time() - t0, 1),
        "epochs": epochs,
        "params": int(sum(v.size for v in model.p.values())),
        "holdoutNodes": int(len(hy_node)),
        "holdoutEdges": int(len(hy_edge)),
        "finalLoss": losses[-1],
        "lossCurve": losses[::max(1, len(losses) // 40)],
        "labels": "simulator weak supervision · reachable within 5h (binarised)",
        "split": "topology-level holdout (unseen estate)",
        "temperature": round(model.temp, 2),
        "calibrationSet": "unseen estate, not the holdout",
        "validationLoss": round(best_val, 4),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **model.state(), temperature=model.temp)
    out_path.with_suffix(".json").write_text(json.dumps(metrics, indent=1))
    return metrics


def load(path: str | Path) -> GraphSAGE:
    model = GraphSAGE()
    data = np.load(path)
    if "temperature" in data.files:
        model.temp = float(data["temperature"])
    model.load_state({k: data[k] for k in data.files if k != "temperature"})
    return model


def prior_for(model: GraphSAGE, estate: Estate) -> tuple[dict, dict, dict]:
    """Absolute GNN priors for nodes and edges, used to sharpen simulation."""
    t = build_tensors(estate)
    nodes_p, edges_p = model.predict(t)
    node_prior = {t["nodes"][i]: float(nodes_p[i]) for i in range(len(nodes_p))}
    edge_prior = {estate.edges[i].key: float(edges_p[i]) for i in range(len(edges_p))}
    base_edge = {e.key: e.p for e in estate.edges}
    return node_prior, edge_prior, base_edge
