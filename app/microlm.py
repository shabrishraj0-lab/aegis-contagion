"""CONTAGION-LM: a decoder-only transformer written and trained from scratch.

No pretrained weights, no tokenizer download, no inference API.  A word-level
causal transformer (embedding + positional encoding, pre-LayerNorm blocks,
multi-head causal self-attention, GELU MLP, tied output projection) with the
forward pass and every gradient implemented in NumPy.

It is deliberately small.  Its job is to draft the prose of an incident memo;
the grounded synthesiser and the numeric validator decide what is true.  That
division is the product's safety argument, not a limitation to hide.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class Tokenizer:
    def __init__(self, vocab: list[str] | None = None) -> None:
        self.itos = vocab or ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def fit(cls, docs: list[str], max_vocab: int = 900) -> "Tokenizer":
        from collections import Counter
        counts = Counter()
        for doc in docs:
            counts.update(doc.split())
        words = [w for w, _ in counts.most_common(max_vocab - 4)]
        return cls(["<pad>", "<bos>", "<eos>", "<unk>"] + words)

    def encode(self, text: str, seq: int | None = None) -> list[int]:
        ids = [BOS] + [self.stoi.get(w, UNK) for w in text.split()] + [EOS]
        if seq:
            ids = ids[:seq] + [PAD] * max(0, seq - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids
                        if i not in (PAD, BOS, EOS))

    def state(self) -> list[str]:
        return self.itos


class NanoGPT:
    def __init__(self, vocab: int, seq: int = 28, d: int = 96,
                 layers: int = 3, heads: int = 4, seed: int = 5) -> None:
        self.V, self.T, self.d, self.L, self.H = vocab, seq, d, layers, heads
        self.hd = d // heads
        rng = np.random.default_rng(seed)

        def mat(*shape, scale=None):
            return (rng.standard_normal(shape) *
                    (scale if scale else 0.02)).astype(np.float32)

        p = {
            "E": mat(vocab, d, scale=0.03),
            "P": mat(seq, d, scale=0.01),
            "lnf_g": np.ones(d, dtype=np.float32),
            "lnf_b": np.zeros(d, dtype=np.float32),
        }
        for l in range(layers):
            for name in ("ln1_g", "ln2_g"):
                p[f"{name}{l}"] = np.ones(d, dtype=np.float32)
            for name in ("ln1_b", "ln2_b"):
                p[f"{name}{l}"] = np.zeros(d, dtype=np.float32)
            for w in ("Wq", "Wk", "Wv", "Wo"):
                p[f"{w}{l}"] = mat(d, d, scale=0.04)
            p[f"W1{l}"] = mat(d, 4 * d, scale=0.04)
            p[f"b1{l}"] = np.zeros(4 * d, dtype=np.float32)
            p[f"W2{l}"] = mat(4 * d, d, scale=0.04)
            p[f"b2{l}"] = np.zeros(d, dtype=np.float32)
        self.p = p
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.step = 0

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _ln(x, g, b, eps=1e-5):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + eps)
        xh = (x - mu) * inv
        return xh * g + b, (xh, inv, g)

    @staticmethod
    def _ln_back(dy, cache, g):
        xh, inv, _ = cache
        dxh = dy * g
        dx = inv * (dxh - dxh.mean(axis=-1, keepdims=True)
                    - xh * (dxh * xh).mean(axis=-1, keepdims=True))
        dg = (dy * xh).sum(axis=(0, 1))
        db = dy.sum(axis=(0, 1))
        return dx, dg, db

    @staticmethod
    def _gelu(x):
        c = math.sqrt(2.0 / math.pi)
        u = c * (x + 0.044715 * x ** 3)
        t = np.tanh(u)
        return 0.5 * x * (1 + t)

    @staticmethod
    def _gelu_back(dy, x):
        c = math.sqrt(2.0 / math.pi)
        u = c * (x + 0.044715 * x ** 3)
        t = np.tanh(u)
        du = c * (1 + 3 * 0.044715 * x ** 2)
        return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du)

    def _attn(self, x, l):
        p = self.p
        B, T, d = x.shape
        q = x @ p[f"Wq{l}"]
        k = x @ p[f"Wk{l}"]
        v = x @ p[f"Wv{l}"]
        qh = q.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)
        kh = k.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)
        vh = v.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)
        scores = np.einsum("bhtd,bhsd->bhts", qh, kh) / math.sqrt(self.hd)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask[None, None], -1e9, scores)
        scores = scores - scores.max(axis=-1, keepdims=True)
        att = np.exp(scores)
        att /= att.sum(axis=-1, keepdims=True)
        out = np.einsum("bhts,bhsd->bhtd", att, vh)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, d)
        y = out @ p[f"Wo{l}"]
        return y, (x, qh, kh, vh, att, out, q, k, v)

    # -------------------------------------------------------------- forward
    def forward(self, ids):
        B, T = ids.shape
        p = self.p
        x = p["E"][ids] + p["P"][:T][None, :, :]
        caches = []
        for l in range(self.L):
            x1, ln1 = self._ln(x, p[f"ln1_g{l}"], p[f"ln1_b{l}"])
            a, ac = self._attn(x1, l)
            x = x + a
            x2, ln2 = self._ln(x, p[f"ln2_g{l}"], p[f"ln2_b{l}"])
            h_pre = x2 @ p[f"W1{l}"] + p[f"b1{l}"]
            h = self._gelu(h_pre)
            f = h @ p[f"W2{l}"] + p[f"b2{l}"]
            x = x + f
            caches.append((ln1, ac, ln2, x1, x2, h_pre, h, x))
        hf, lnf = self._ln(x, p["lnf_g"], p["lnf_b"])
        logits = hf @ p["E"].T
        return logits, (ids, caches, hf, lnf, h_pre)

    def loss(self, ids, targets, mask=None):
        logits, cache = self.forward(ids)
        B, T, V = logits.shape
        logits = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        if mask is None:
            mask = np.ones((B, T), dtype=np.float32)
        denom = max(1.0, mask.sum())
        nll = -np.log(np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0] + 1e-9)
        loss = float((nll * mask).sum() / denom)
        dlogits = probs.copy()
        np.put_along_axis(dlogits, targets[..., None],
                          np.take_along_axis(dlogits, targets[..., None], -1) - 1,
                          axis=-1)
        dlogits *= mask[..., None] / denom
        return loss, dlogits, cache

    # ------------------------------------------------------------- backward
    def backward(self, dlogits, cache):
        ids, caches, hf, lnf, _ = cache
        p = self.p
        grads = {}
        B, T = ids.shape
        dE = np.zeros_like(p["E"])
        dhf = dlogits @ p["E"]
        np.add.at(dE, ids.reshape(-1), dhf.reshape(-1, self.d))
        dx, dg, db = self._ln_back(dhf, lnf, p["lnf_g"])
        grads["lnf_g"], grads["lnf_b"] = dg, db
        for l in reversed(range(self.L)):
            ln1, ac, ln2, x1, x2, h_pre, h, x_after = caches[l]
            # MLP branch
            dh = dx @ p[f"W2{l}"].T
            grads[f"W2{l}"] = h.reshape(-1, 4 * self.d).T @ dx.reshape(-1, self.d)
            grads[f"b2{l}"] = dx.sum(axis=(0, 1))
            dh_pre = self._gelu_back(dh, h_pre)
            grads[f"W1{l}"] = x2.reshape(-1, self.d).T @ dh_pre.reshape(-1, 4 * self.d)
            grads[f"b1{l}"] = dh_pre.sum(axis=(0, 1))
            dx2 = dh_pre @ p[f"W1{l}"].T
            dx_ln2, dg2, db2 = self._ln_back(dx2, ln2, p[f"ln2_g{l}"])
            grads[f"ln2_g{l}"], grads[f"ln2_b{l}"] = dg2, db2
            dx = dx + dx_ln2
            # attention branch
            x_in, qh, kh, vh, att, out, q, k, v = ac
            dO = dx @ p[f"Wo{l}"].T
            grads[f"Wo{l}"] = out.reshape(-1, self.d).T @ dx.reshape(-1, self.d)
            dO = dO.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3)
            datt = np.einsum("bhtd,bhsd->bhts", dO, vh)
            dvh = np.einsum("bhts,bhtd->bhsd", att, dO)
            dscore = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
            dqh = np.einsum("bhts,bhsd->bhtd", dscore, kh) / math.sqrt(self.hd)
            dkh = np.einsum("bhts,bhtd->bhsd", dscore, qh) / math.sqrt(self.hd)
            dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            xin_flat = x1.reshape(-1, self.d)
            grads[f"Wq{l}"] = xin_flat.T @ dq.reshape(-1, self.d)
            grads[f"Wk{l}"] = xin_flat.T @ dk.reshape(-1, self.d)
            grads[f"Wv{l}"] = xin_flat.T @ dv.reshape(-1, self.d)
            dx1 = dq @ p[f"Wq{l}"].T + dk @ p[f"Wk{l}"].T + dv @ p[f"Wv{l}"].T
            dx_ln1, dg1, db1 = self._ln_back(dx1, ln1, p[f"ln1_g{l}"])
            grads[f"ln1_g{l}"], grads[f"ln1_b{l}"] = dg1, db1
            dx = dx + dx_ln1
        grads["E"] = dE
        # positional gradient: sum over batch
        pos = np.zeros_like(p["P"])
        pos[:T] = dx.sum(axis=0)
        grads["P"] = pos
        return grads

    # ------------------------------------------------------------------ adam
    def adam(self, grads, lr=3e-3, clip=1.0):
        self.step += 1
        b1, b2, eps = 0.9, 0.98, 1e-8
        norm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, clip / (norm + 1e-9))
        for k, g in grads.items():
            g = g * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            mh = self.m[k] / (1 - b1 ** self.step)
            vh = self.v[k] / (1 - b2 ** self.step)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    # --------------------------------------------------------------- sample
    def sample(self, tok: Tokenizer, prompt: str = "outbreak memo .",
               max_new: int = 34, temperature: float = 0.72,
               top_k: int = 24, seed: int = 3) -> str:
        rng = np.random.default_rng(seed)
        ids = tok.encode(prompt)[:-1][: self.T]
        for _ in range(max_new):
            window = ids[-self.T:]
            arr = np.array([window], dtype=np.int64)
            logits, _ = self.forward(arr)
            logit = logits[0, -1].astype(np.float64) / max(0.15, temperature)
            logit[PAD] = logit[BOS] = -1e9
            k = min(top_k, len(logit))
            top = np.argpartition(-logit, k - 1)[:k]
            probs = np.exp(logit[top] - logit[top].max())
            probs /= probs.sum()
            nxt = int(rng.choice(top, p=probs))
            ids.append(nxt)
            if nxt == EOS:
                break
        return tok.decode(ids[len(tok.encode(prompt)[:-1]):])

    # ------------------------------------------------------------------- io
    def state(self) -> dict:
        return {k: v for k, v in self.p.items()}

    def load_state(self, state: dict) -> None:
        for k, v in state.items():
            if k in self.p and getattr(v, "shape", None) == self.p[k].shape:
                self.p[k] = np.asarray(v, dtype=np.float32)

    @property
    def params(self) -> int:
        return int(sum(v.size for v in self.p.values()))


def train(out_base: str | Path, steps: int = 340, quick: bool = False,
          seed: int = 5) -> dict:
    from .corpus import build_corpus
    t0 = time.time()
    docs = build_corpus(1600 if not quick else 500, seed=77)
    tok = Tokenizer.fit(docs, max_vocab=700)
    seq = 30
    data = np.array([tok.encode(d, seq) for d in docs], dtype=np.int64)
    rng = np.random.default_rng(seed)
    model = NanoGPT(len(tok.itos), seq=seq, d=96, layers=3, heads=4, seed=seed)
    batch = 24
    losses = []
    n_steps = steps if not quick else 60
    for step in range(n_steps):
        idx = rng.integers(0, len(data), size=batch)
        chunk = data[idx]
        ids, targets = chunk[:, :-1], chunk[:, 1:]
        mask = (targets != PAD).astype(np.float32)
        loss, dlogits, cache = model.loss(ids, targets, mask)
        grads = model.backward(dlogits, cache)
        model.adam(grads, lr=4e-3 if step < n_steps * 0.7 else 1.2e-3)
        losses.append(round(loss, 4))
        if step % 40 == 0:
            print(f"  lm step {step}/{n_steps} loss {loss:.3f}")
    sample = model.sample(tok, "outbreak memo . estate meridian health .", seed=9)
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(base, **model.state())
    metrics = {
        "params": model.params,
        "vocab": len(tok.itos),
        "layers": model.L,
        "heads": model.H,
        "dModel": model.d,
        "seq": seq,
        "steps": n_steps,
        "docs": len(docs),
        "finalLoss": losses[-1],
        "initialLoss": losses[0],
        "perplexity": round(math.exp(min(9.9, losses[-1])), 2),
        "trainSeconds": round(time.time() - t0, 1),
        "lossCurve": losses,
        "sample": sample,
        "tokenizer": tok.state(),
    }
    base.with_suffix(".json").write_text(json.dumps(metrics, indent=1))
    return metrics


def load(base: str | Path) -> tuple[NanoGPT, Tokenizer, dict]:
    base = Path(base)
    meta = json.loads(base.with_suffix(".json").read_text())
    tok = Tokenizer(meta["tokenizer"])
    model = NanoGPT(len(tok.itos), seq=meta["seq"], d=meta["dModel"],
                    layers=meta["layers"], heads=meta["heads"])
    data = np.load(base)
    model.load_state({k: data[k] for k in data.files})
    return model, tok, meta
