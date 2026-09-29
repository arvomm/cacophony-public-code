# The supervenience Bayesian network — companion code

Companion code for the paper's Bayesian approach to assessing a system's
consciousness (Section 7, *"Indicators, evidence, and the attribution of
consciousness to AI systems"*). It implements the supervenience Bayesian
network over the five-level hierarchy of functional descriptions, scores
evidence vectors of indicator activations, and aggregates per-level posteriors
into an overall credence.

## Start here

**Open the notebook: [`notebooks/scoring.ipynb`](notebooks/scoring.ipynb).**
It is the main thing to read. GitHub displays it directly in your browser, with
all results and figures already included, so there is nothing to install and
nothing to run. You can skip the code cells and read straight down: the text and
figures walk through the model, the five example systems, and how the overall
credence is computed.

Want to play with it instead? Use the
[live interactive explorer](https://ai-cognition.org/cacophony-tool/): click indicators, drag the sliders, and
watch every posterior update live. Nothing to install.

Everything below is background and setup for people who want to run or modify the code.

## The model

For each Level *i* ∈ {1,…,5} of the hierarchy — 1 Behavioural, 2 Computational
functional, 3 Intrinsic causal-structure functional, 4 Organismic, 5
Organism-environment (4E) — a Boolean node **C_i** represents *"the system
instantiates, at Level i, the organisation that Level-i theories take to
suffice for consciousness"*. The nodes form a directed chain rooted at the most
fine-grained level,

```
C5  →  C4  →  C3  →  C2  →  C1
```

and each of the 37 indicators **I_{i,k}** is a child of its level node C_i.
Every edge is a noisy channel characterised by a true-positive and a
false-positive rate:

| parameter | meaning | default |
|---|---|---|
| `alpha_C = P(C_i=1 \| C_{i+1}=1)` | supervenience strength | prior `Beta(8,2)` (mean 0.8); **strict = 1** |
| `beta_C = P(C_i=1 \| C_{i+1}=0)`  | edge false-positive rate ("leak") | prior `Beta(2,8)` (mean 0.2) |
| `pi = P(C5=1)`                    | root prior | `Beta(1,1)`, with a four-shape sensitivity sweep |
| `alpha_j, beta_j`                 | per-indicator TPR / FPR | derived from the DCM's *support* / *demandingness* labels (`dcm priors/mapping.md`) |

Two regimes:

- **Strict Supervenience Model** (paper §7.2.1, **Condition 2A**: `alpha_C = 1`).
  Consciousness at a finer-grained level deterministically guarantees it at
  every coarser level, only the six "staircase" configurations carry mass, and
  the **monotonic belief property** holds: `P(C1|e) ≥ P(C2|e) ≥ … ≥ P(C5|e)`.
- **Generalised model** (paper §7.2.2, **Condition 2B**: `alpha_C ≤ 1`) — the
  default. The deterministic constraint becomes a probabilistic association
  (accommodating e.g. the locked-in patient: a Level 2 activation without the
  Level 1 behavioural activation), and all 2⁵ = 32 configurations carry mass.

Given the per-level posteriors, the **overall credence** that a system is
conscious (paper §7.3) is the theoretical-credence-weighted average over the
critical level L*:

```
P(C | E) = Σ_i P(L* = i) · P(C_i | E)
```

The five worked example systems — Human, Fly, LLM-optimist, LLM-sceptic,
Thermostat (paper §7.4) — use evidence vectors fabricated by the authors to
illustrate model behaviour; they are not derived from data.

## Layout

- `dcm_lab/indicators.py` — loads the 37 indicators and their (alpha, beta)
  priors from `dcm priors/Indicators Caco Nodes DCM priors.csv`.
- `dcm_lab/truth.py` — network constants, edge priors, `make_prior_config`
  (with the `strict` toggle).
- `dcm_lab/score.py` — `prior_predictive_score()` (per-level posteriors
  `P(C_i=1|e)` over all 32 configurations), `overall_credence()` (§7.3),
  `prior_C_marginals()`, `prior_depth_distribution()`, `hdi_from_samples()`.
- `dcm priors/` — the indicator CSV plus `mapping.md` / `mapping.py`
  documenting how DCM (support, demandingness) labels become (alpha, beta).
- `notebooks/scoring.ipynb` — pedagogical walk-through: priors, strict vs
  generalised, sensitivity sweeps, the five illustrative systems, overall credence.

## Running

```
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/jupyter notebook notebooks/scoring.ipynb     # explore / edit interactively
```

To re-execute the notebook headlessly (this also writes figures to `outputs/figures/`):

```
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/scoring.ipynb
```

Everything executes in seconds — inference is exact enumeration over the 32
configurations, vectorised over Monte-Carlo draws of the priors; there is no
MCMC.

## Notes

- The reuse of DCM support/demandingness parameters is a first approximation
  to illustrate model behaviour; see the paper for the caveats.
