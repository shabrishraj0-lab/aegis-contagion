# AEGIS // CONTAGION

**Simulation-driven credential containment with a Hindsight memory that compounds.**

> The scanner finds a key. AEGIS finds the outbreak, prices it, cuts it, and remembers what worked.

AEGIS Contagion is a working prototype for the *AI Agents That Learn Using Hindsight* hackathon. It
models a heterogeneous identity estate — credentials, workloads, repositories, CI pipelines, cloud
roles, data stores, identity providers and AI endpoints — then simulates how one leaked secret becomes
a crown-jewel breach. The agent plans a constrained ring-vaccination portfolio, narrates it with a
language model it trained from scratch, executes a simulated response choreography, and writes the
episode back to Hindsight. The next incident recalls that experience and plans differently.

The topology is synthetic. The algorithms, models, memory loop and optimisation are real.

---

## Why this problem

Every security scanner can tell you a key leaked. Almost none can tell you:

- which production identities that key can actually reach;
- which chain is most likely, and how many minutes it takes;
- what the breach would cost, conditional on that chain;
- which **portfolio** of revocations, session kills, OAuth-grant removals and scope reductions cuts
  the most expected loss without taking the business down;
- and what the team learned from the last time this happened.

That last question is the product. Stateless incident response repeats its mistakes. AEGIS is built
around the opposite premise: **the response plan should start from experience, not zero.**

---

## The loop

```
signal
  │
  ├─ recall ──────────────► Hindsight bank: episodes · facts · directives · edge posteriors
  │                              │
  ├─ beliefs ◄───────────────────┘   edge multipliers, relation multipliers, action priors,
  │                                   recalled portfolios, focus targets, analyst directives
  │
  ├─ simulate ───────────► vectorised Monte Carlo outbreak over the identity graph
  │                         R₀ · time-to-crown-jewel · P50/P90/P99 loss · attack paths
  │
  ├─ plan ───────────────► CELF greedy + exhaustive shortlist under a disruption budget
  │                         memory-seeded portfolios · learned choke points · directives applied
  │
  ├─ narrate ────────────► CONTAGION-LM draft → numeric validator → deterministic grounded memo
  │
  ├─ execute ────────────► dual-key replacement, canary, health check, revoke, rollback
  │                         human approval · idempotency key · audit trail
  │
  └─ retain ─────────────► episode + facts + directives + structural posteriors
                             └──────────────► next incident starts warmer
```

### Seven stages

1. **Ingest / graph** — 80 nodes and 143 evidenced edges per estate, 12 relation types, 5 critical
   assets. Every edge carries a probability, a confidence, a source (`observed` / `inferred`) and a
   last-verified TTL. Three seeded estates: Meridian Health, Northwind Payments, Orbital Logistics.
2. **Recall** — Hindsight returns narrative records; the local mirror implements the same four-channel
   retrieval contract.
3. **Believe** — records become numbers: edge transmission posteriors, relation-level corrections,
   action friction priors, recalled portfolios and analyst directives.
4. **Simulate** — vectorised NumPy Monte Carlo. Geometric time-to-success per edge, exponential
   detection dwell, Bellman-Ford relaxation over infection times, attacker policy (`balanced`, `loud`,
   `quiet`), and a residual-path reservoir from every foothold the attacker plausibly already has.
5. **Plan** — a constrained maximum-coverage problem. Cheap reversible cuts first, rotations last.
   A naive "rotate the three riskiest credentials" baseline is computed for comparison.
6. **Narrate** — a decoder-only transformer writes a draft; every numeric token is checked against the
   simulation. If the draft is not fully grounded it is rejected and the deterministic synthesiser
   writes the memo. The model never sources a number.
7. **Retain** — the episode, its facts, analyst verdict, structural posteriors and a reusable action
   portfolio are written back. The next plan can start from them.

---

## Memory: the star

### What is stored

| Type | Example | Used for |
|---|---|---|
| **Episode** | `INC-1041 · Meridian · CI secret · plan revoke×3 + remove_oauth · coverage 96.9% · verdict contained` | case-based portfolio replay |
| **Fact** | `The chain gh-actions-release → release-prod → AWSProdDeployRole carried 12% of outbreaks and $5.0M conditional loss in 57 minutes.` | semantic recall, evidence |
| **Directive** | `Analyst ...: hold automated rotation of gh-actions-release until consumers are re-inventoried.` | constraints on the next plan |
| **Edge posterior** | `edge(gh-actions-release → release-prod): Beta(alive, dead)` | sharper next forecast |
| **Relation posterior** | `can_deploy: Beta(alive, dead)` | relation-level correction |
| **Action prior** | `revoke_sessions: coverage-per-disruption` | lower operational friction for proven actions |
| **Portfolio** | `actions + targets + realised coverage` | warm-start candidate in the optimiser |
| **Hotspot focus** | nodes that repeatedly appeared on dominant chains | memory-directed search attention |

### Four-channel recall

The local Hindsight mirror implements:

1. **Dense** — deterministic hashed character/word n-gram embedding (no download, no network).
2. **Keyword** — BM25 over the memory text.
3. **Graph** — entity expansion through the memory's own entity graph.
4. **Temporal** — recency decay.

Channels are fused with reciprocal-rank fusion, then reranked by **MemoryRank**, a small NumPy MLP
trained online from analyst feedback. The same interface is backed by the official
`hindsight-client` when cloud credentials are configured.

### What actually changes in the plan

The warm plan is not a cosmetic overlay. Memory can:

- **lower the modelled disruption** of an action that worked before, allowing another intervention to
  fit inside the budget;
- **add recalled targets** to the exhaustive shortlist, so the search looks where experience says the
  choke points are;
- **replay a whole portfolio** from a prior episode on the same estate as a candidate;
- **exclude a rotation** the analyst explicitly asked to hold;
- **sharpen edge/relation transmission probabilities** from observed outbreak outcomes, with shrinkage
  so one noisy episode cannot hijack the forecast.

### Hindsight setup

The demo works with zero credentials using the local mirror. To use Hindsight Cloud:

```bash
export HINDSIGHT_BASE_URL="https://api.hindsight.vectorize.io"
export HINDSIGHT_API_KEY="your-key"
export HINDSIGHT_BANK="aegis-contagion"      # optional; this is the default bank id
```

When the two required variables are present, `MemoryBank` instantiates the official
`hindsight-client` and uses it for retain/recall/reflect, while keeping numeric posteriors locally.
Cloud failures are logged and the local mirror continues to serve, so a venue network cannot kill the
demo.



---

## Models built here

No pretrained weights. No inference API. Both models are implemented and trained in NumPy, with
manual forward and backward passes.

### CONTAGION-GNN

A relation-aware GraphSAGE for the identity graph.

- 21 node features: kind one-hot, privilege, scopes, exposure, MFA, credential age, criticality,
  value, in/out degree, AI endpoint, credential and repo/pipeline flags.
- 10 edge features: 6 relation-group one-hots, base probability, time cost, confidence, inferred flag.
- Two message-passing layers, 48 hidden units, relation-specific weight matrices.
- Two heads: `P(node reached within 5h)` and `P(edge transmits)`.
- Training labels: simulator weak supervision (stated as such), binarised at 10% / 5%.
- Topology-level holdout: an estate family the model never trained on. Temperature scaling is fitted
  on a separate calibration estate, not the holdout.

Measured holdout (current shipped weights):

| Metric | Node head | Edge head |
|---|---|---|
| Average precision | **0.828** | **0.880** |
| ECE (10 bins) | **0.130** | 0.083 |
| Brier | 0.166 | 0.131 |
| Base rate | 0.425 | 0.375 |

26,786 parameters · 160 epochs · ~1s training on this machine · temperature 2.0. The labels are
simulator-generated; this is a learned surrogate of the simulator's transmission process, not a claim
about real breach history.

### CONTAGION-LM

A decoder-only transformer trained from scratch on 1,603 synthetic incident memos.

- 3 layers, 4 heads, d_model 96, context 30, vocab 258, **362,208 parameters**.
- 60 training steps, 8.8s on this machine; final training loss 1.07, perplexity 2.92.
- It writes domain-shaped but imperfect prose. That is deliberate: the raw draft is displayed as
  **untrusted**. Every number in it is extracted and checked against the simulator's admissible set.
  In the shipped run the draft had 0/4 grounded numeric claims, so the deterministic grounded memo was
  used. The final memo re-checks each claim against the simulation (for example 8/8 or 9/9 present).

This is a small research model, not GPT-4. The safety story is the validator and the grounded
synthesiser, not the fluency of the draft.

---

## Simulation and planner

### Monte Carlo

For each trial:

- each edge succeeds after `Geometric(p) × t` minutes;
- each infected node has a detection hazard (exponential dwell, kind-specific);
- infection times are relaxed with a vectorised Bellman-Ford, so cycles and multiple paths are handled;
- the attacker follows a policy that changes speed/stealth;
- trial loss is the conditional five-component loss of the first critical asset reached.

Outputs: R₀, reach probability, time-to-crown-jewel quantiles, P50/P90/P99 loss, dominant paths,
edge fire rates, next-hop marginals, credential spread, and the infection curve used by the scrubber.
The **residual-path reservoir** enumerates up to 64 alternative chains from every likely-compromised
credential to every critical asset, so a plan must cut a set of chains rather than one heroic vertex.

### Ring-vaccination optimiser

- Action menu: deny ASN, revoke sessions, remove OAuth grant, step-up MFA, reduce scope, rotate,
  disable. Rotations are durable but expensive; they are ordered last.
- Greedy selection with lazy evaluation (CELF) plus exhaustive search over the shortlisted actions.
- Constraints: disruption budget (default 8%) and cost budget. Human approval for destructive actions.
- Baseline: rotate the three highest-degree credentials.
- Memory-conditioned: learned friction, recalled portfolios, focus targets and directives.

---

## Measured learning result

The evaluation is deliberately conservative:

- Each incident builds a **cold plan** from a memory-free forecast and a **warm plan** from the
  memory-conditioned forecast.
- Both plans are scored on a **third, memory-free held-out reservoir** that neither planner saw.
- Episode 1 has no memory available, so it is excluded from the headline average. Counting its random
  forecast gap as “learning” would be dishonest.
- An incident with no residual chain to a critical asset is excluded from coverage averages rather
  than scored as 0%.

Current six-episode replay (900 outbreaks per plan, 8% disruption budget):

| # | Incident | Reachable paths | Recalled | Cold | Warm | Δ |
|---|---|---|---:|---:|---:|---:|
| 1 | Leaked AI key | 21 | 0 | 71.5% | 94.9% | cold start · excluded |
| 2 | CI secret exposure | 16 | 5 | 83.1% | 97.6% | **+14.4pp** |
| 3 | OAuth grant abuse | 15 | 6 | 86.7% | 86.6% | −0.1pp |
| 4 | Identity admin token | 5 | 6 | 99.6% | 99.6% | 0.0pp |
| 5 | Deploy role compromise | 20 | 6 | 92.8% | 92.5% | −0.3pp |
| 6 | Package token theft | 0 | 6 | — | — | no reachable critical asset |

**Memory-eligible mean: 90.6% cold → 94.1% warm, +3.5pp held-out coverage.**
Naive rotation baseline on the same episodes: 17.5%.

The honest interpretation: the constrained optimiser is already strong, so memory does not magically
double coverage. Its largest effect is forecast correction when an estate repeats (+14.4pp on the
second Meridian incident), plus analyst constraints, recalled case portfolios and lower action
friction. Coverage alone understates the behavioural change; the memory tab shows the records, the
directives and the exact edge/relation posteriors that were applied.

These numbers are stochastic. Re-running the replay changes them; the code writes the new series to
`data/campaign.json` and the UI reads it directly.

---

## Run it

Requirements: Python 3.11+ (tested on 3.13.7), Node is not required to run the app.

```bash
cd contagion
pip install -r requirements.txt
python run.py                 # http://127.0.0.1:8765
python run.py --no-browser    # if you do not want the browser to open
```

The first boot loads the shipped `data/gnn.npz` and `data/lm.npz`. If a model file is missing, the
agent trains it synchronously before serving.

The console is vanilla HTML/CSS/ES modules with custom Canvas renderers. There is no frontend build
step. Keyboard shortcuts: `1`–`5` switch tabs, `M` toggles memory, `E` executes, `Space` replays the
outbreak clock, `R` reruns the selected scenario, `Cmd/Ctrl+K` opens the command palette.

### Demo script (60–90 seconds)

1. Open the console. The boot log shows the graph, GNN, LM and Hindsight bank loading.
2. Run **Leaked AI key**. Point at R₀, time-to-crown-jewel, P50/P90 loss and the dominant chain.
3. Open the **plan** tab. Show the portfolio, the coverage frontier, and the three held-out bars:
   naive rotation vs cold agent vs Hindsight agent.
4. Click **Evidence** on a path. Every hop shows its relation, probability, confidence and source.
5. Execute the plan. The modal asks for an analyst verdict. Show the dual-key choreography, health
   check and audit trail.
6. Open **Memory**. Search `which cut vertex broke the release chain?`. Show episodes, facts,
   directives, the memory constellation, and the edge/relation posteriors.
7. Open **Model**. Show node AP 0.828, edge AP 0.880, ECE 0.13, the training-loss curve, the LM card,
   and the numeric validator rejecting the raw draft.
8. Open **Eval**. Replay six incidents. The ledger shows the cold and warm held-out scores; the
   memory-eligible mean is +3.5pp, with a +14.4pp repeat on Meridian.
9. Toggle **Memory OFF**, rerun the same scenario, and watch the badge change from `LEARNED` to
   `COLD`, the directives disappear and the plan start from the generic playbook.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/bootstrap` | estates, scenarios, incidents, memory, model metadata |
| `GET` | `/api/graph?estate=meridian` | full graph with provenance |
| `POST` | `/api/run` | simulate + plan + narrate; body includes `useMemory`, `budget`, `trials`, `horizon`, `policy` |
| `POST` | `/api/execute` | simulated choreography + retain episode |
| `POST` | `/api/campaign` | sequential six-incident replay |
| `GET` | `/api/memory?q=...` | four-channel recall |
| `GET` | `/api/memory/stats` | bank stats, records, directives, episodes, action stats |
| `POST` | `/api/memory/reset` | delete the bank and return to cold start |
| `GET` | `/api/eval` | model card and campaign results |
| `GET` | `/api/incident/{id}` | full incident payload |

---

## What is real vs synthetic

**Real:** the graph algorithms, Monte Carlo simulator, constrained optimiser, GraphSAGE forward/backward
passes, transformer forward/backward passes, four-channel retrieval, reciprocal-rank fusion, learned
reranker, numeric validator, response choreography, audit/idempotency model, memory persistence and
the frontend.

**Synthetic:** the estate topology, node names, edge probabilities, dollar values and incident memos.
Labels are simulator-generated weak supervision. Dollar figures are conditional ranges, not accounting
figures. Coverage is residual expected-loss coverage, not a guarantee that no breach occurs.

---

## Honest limitations

- The topology is synthetic; no real customer data or breach history is used.
- The GNN is a surrogate trained on simulator labels. AP 0.828 is measured against those labels, not
  against ground-truth compromises.
- The LM is tiny and often incoherent. It is never trusted for numbers; the deterministic synthesiser
  and validator do the factual work.
- The optimiser is already near-optimal on the small synthetic graph, so memory's coverage lift is
  modest and stochastic. The strongest repeat is +14.4pp; the memory-eligible mean is +3.5pp.
- Hindsight Cloud is implemented behind the official client but was not exercised here because no cloud
  credentials were available. The local mirror is the verified demo path.
- **Band AI desktop:** Band is the hackathon's collaboration surface, but this build does not embed the
  Band SDK or claim a live Band integration. No Band account/API key was available. The incident memo,
  evidence trace and action ledger are structured so they can be pasted into a Band room; a real
  integration would register an external Band agent and publish those fields through the Band Agent API.
- Execution is simulated. No real credential is rotated, revoked or disabled.

---

## Repository layout

```
contagion/
├── run.py                  launch the FastAPI console
├── requirements.txt
├── app/
│   ├── estate.py           synthetic identity estates + graph
│   ├── simulator.py        Monte Carlo + residual paths
│   ├── optimizer.py        action portfolio + CELF + exhaustive search
│   ├── impact.py           five-component conditional loss
│   ├── gnn.py              relation-aware GraphSAGE, NumPy autograd
│   ├── microlm.py          decoder-only transformer, NumPy autograd
│   ├── corpus.py           synthetic incident-memo corpus
│   ├── memory.py           local Hindsight + cloud wrapper + MemoryRank
│   ├── agent.py            recall → simulate → plan → narrate → execute → retain
│   ├── response.py         dual-key choreography + audit
│   └── server.py           FastAPI surface
├── web/
│   ├── index.html          console shell
│   ├── styles.css          forensic dark UI
│   └── js/                 api, charts, graph, ui, app
└── data/
    ├── gnn.npz/.json       trained GNN + metrics
    ├── lm.npz/.json        trained LM + metrics
    ├── memory_bank.json    Hindsight records
    ├── memory_numeric_*.json  posteriors, directives, portfolios
    ├── incidents.json      incident ledger
    └── campaign.json       last replay results
```

---

## Acknowledgements

Built for the **AI Agents That Learn Using Hindsight** hackathon. Memory layer designed around
[Vectorize Hindsight](https://hindsight.vectorize.io/). No pretrained model weights, no inference API,
no frontend framework.
