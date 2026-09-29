"""Structural constants and prior-config helpers for the supervenience model.

This module implements the Bayesian-network structure of Section 7.2 of the
paper ("From supervenience hierarchy to conditional independence"). The five
consciousness nodes C_1 .. C_5 form a directed chain

    C5  ->  C4  ->  C3  ->  C2  ->  C1

where C_i = 1 represents the proposition "the system instantiates, at Level i,
the organisation that Level-i theories take to suffice for consciousness". The
root is C5 (the organism-environment level, the most fine-grained); probability
flows down the chain toward C1 (the behavioural level, the most coarse-grained):
each Level i is a coarse-graining of Level i+1.

Each edge (parent C_{i+1} -> child C_i) is a noisy channel characterised by two
parameters, named to mirror the indicator layer (which also uses alpha = TPR,
beta = FPR):

    alpha_C = P(C_i = 1 | C_{i+1} = 1)   supervenience strength.
    beta_C  = P(C_i = 1 | C_{i+1} = 0)   edge false-positive rate ("leak"):
                                         how often the coarser-level organisation
                                         is present when the finer level is absent.

Two regimes, corresponding to the paper's two models:

  - **Strict Supervenience Model** (paper 7.2.1, Condition 2A): alpha_C = 1.
    Consciousness at a finer-grained level deterministically guarantees
    consciousness at every coarser level above it (C_{i+1} => C_i), so only the
    six "staircase" configurations carry mass and the monotonic belief property
    holds: P(C1|e) >= P(C2|e) >= ... >= P(C5|e). Set strict=True.

  - **Generalised model** (paper 7.2.2, Condition 2B): alpha_C <= 1, here given
    the prior Beta(8, 2) (mean 0.8). The deterministic constraint is weakened to
    a probabilistic association -- e.g. a locked-in patient can have a Level 2
    activation without the Level 1 behavioural activation. All 2**5 = 32
    configurations carry mass. This is the default.

In both regimes beta_C remains a free parameter (prior Beta(2, 8), mean 0.2).

The root C5 carries a Beta prior pi = P(C5 = 1); a four-shape sweep is provided
for sensitivity analysis.
"""
from __future__ import annotations

import numpy as np


L = 5   # number of consciousness levels, fixed by the five-level hierarchy


# Four-shape sensitivity sweep for the root prior pi = P(C5 = 1).
C_PRIOR_SWEEP = [
    ("Beta(1,1)", (1.0, 1.0)),
    ("Beta(2,2)", (2.0, 2.0)),
    ("Beta(6,2)", (6.0, 2.0)),
    ("Beta(2,6)", (2.0, 6.0)),
]

# Default priors on the supervenience-edge parameters (generalised model).
C_ALPHA_PRIOR = (8.0, 2.0)   # alpha_C = P(C_i=1 | C_{i+1}=1), mean 0.8 — supervenience strength
C_BETA_PRIOR  = (2.0, 8.0)   # beta_C  = P(C_i=1 | C_{i+1}=0), mean 0.2 — leak


def make_prior_config(root_prior: tuple,
                      indicator_alphas,
                      indicator_betas,
                      c_alpha_prior: tuple = C_ALPHA_PRIOR,
                      c_beta_prior: tuple = C_BETA_PRIOR,
                      strict: bool = False) -> dict:
    """Build a prior config dict for the supervenience model.

    Parameters
    ----------
    root_prior : (a, b)
        Beta-prior parameters for the root pi = P(C5 = 1).
    indicator_alphas, indicator_betas : 1-D arrays of length N_IND
        Per-indicator TPR / FPR: alpha_j = P(I_j = 1 | C_i = 1) and
        beta_j = P(I_j = 1 | C_i = 0), where C_i is indicator j's parent level.
    c_alpha_prior : (a, b)
        Beta prior on the edge parameter alpha_C = P(C_i=1 | C_{i+1}=1).
        Ignored when `strict` (alpha_C pinned to 1, Condition 2A).
    c_beta_prior : (a, b)
        Beta prior on the edge false-positive rate beta_C = P(C_i=1 | C_{i+1}=0).
        Sampled in both regimes.
    strict : bool
        If True, the Strict Supervenience Model (Condition 2A: alpha_C = 1); only
        the six staircase configurations survive. If False (default), the
        generalised model (Condition 2B: alpha_C ~ Beta prior), where all
        2**5 = 32 configurations carry mass.
    """
    if not (isinstance(root_prior, tuple) and len(root_prior) == 2):
        raise ValueError(f"root_prior must be (a, b); got {root_prior!r}")
    return {
        "root":            root_prior,
        "c_alpha":         c_alpha_prior,
        "c_beta":          c_beta_prior,
        "strict":          bool(strict),
        "indicator_alpha": np.asarray(indicator_alphas, dtype=float),
        "indicator_beta":  np.asarray(indicator_betas,  dtype=float),
    }
