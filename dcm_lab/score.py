"""Single-case scoring under the supervenience Bayesian network (paper Section 7).

The network (Section 7.2) is the directed chain

    C5  ->  C4  ->  C3  ->  C2  ->  C1          (root = C5)

with each indicator I_{i,k} a child of its level node C_i. The joint distribution
factorises as

    P(C1..C5, E) = P(C5) * prod_i P(C_i | C_{i+1}) * prod_{i,k} P(I_{i,k} | C_i)

Edge parameters (shared across the four C-edges, mirroring the indicator layer):

    alpha_C = P(C_i = 1 | C_{i+1} = 1)    supervenience strength
    beta_C  = P(C_i = 1 | C_{i+1} = 0)    edge false-positive rate ("leak")

Indicator parameters (fixed per indicator, derived from the DCM's support and
demandingness labels; see `dcm priors/mapping.md`):

    alpha_j = P(I_j = 1 | parent C_i = 1)   indicator TPR
    beta_j  = P(I_j = 1 | parent C_i = 0)   indicator FPR

Two regimes (set by prior_config['strict']):

  - **Generalised model** (default; paper 7.2.2, Condition 2B): alpha_C and
    beta_C both sampled from Beta priors. All 2**5 = 32 configurations of
    (C1..C5) carry mass; a coarser level can be present without the finer
    one below it (e.g. behaviour without the computational organisation).
  - **Strict Supervenience Model** (paper 7.2.1, Condition 2A): alpha_C = 1,
    so C_{i+1} => C_i and only the 6 "staircase" configurations survive. The
    per-level posteriors then obey the monotonic belief property
    P(C1|e) >= P(C2|e) >= ... >= P(C5|e).

We always enumerate all 32 configurations; under `strict` the non-staircase
configurations get log-prior ~ log(1 - alpha_C) = log(0) (clamped), i.e. ~zero
mass, so a single code path covers both regimes. Everything is vectorised over
the n_draws prior samples; runtime is milliseconds. Given the per-level
posteriors, `overall_credence` computes the paper's Section 7.3 weighted
average P(C|E) = sum_i P(L* = i) * P(C_i | E).
"""
from __future__ import annotations

import itertools

import numpy as np

from .truth import L


def _safe_log(x):
    return np.log(np.maximum(x, 1e-300))


def hdi_from_samples(samples, prob: float = 0.89, axis: int = 0):
    """Highest density interval (HDI) of an empirical sample."""
    s = np.asarray(samples)
    if s.ndim == 1:
        sorted_s = np.sort(s)
        n = sorted_s.shape[0]
        k = int(np.ceil(prob * n))
        if k >= n:
            return float(sorted_s[0]), float(sorted_s[-1])
        widths = sorted_s[k:] - sorted_s[: n - k]
        i = int(np.argmin(widths))
        return float(sorted_s[i]), float(sorted_s[i + k])
    if s.ndim != 2:
        raise ValueError("hdi_from_samples supports 1-D or 2-D arrays only")
    if axis == 0:
        sorted_s = np.sort(s, axis=0)
        n = sorted_s.shape[0]
    else:
        sorted_s = np.sort(s, axis=1).T
        n = sorted_s.shape[0]
    k = int(np.ceil(prob * n))
    if k >= n:
        return sorted_s[0].copy(), sorted_s[-1].copy()
    widths = sorted_s[k:] - sorted_s[: n - k]
    i = np.argmin(widths, axis=0)
    cols = np.arange(sorted_s.shape[1])
    lo = sorted_s[i, cols]
    hi = sorted_s[i + k, cols]
    return lo, hi


def _enumerate_configs(L_: int = L):
    """The 6 "staircase" configurations (depth d in 0..L), the only ones with
    non-zero mass under the Strict Supervenience Model."""
    return [tuple(1 if k < d else 0 for k in range(L_)) for d in range(L_ + 1)]


def _enumerate_configs_full(L_: int = L):
    """All 2**L configurations (c1..cL), index 0..L-1 for levels 1..L."""
    return [tuple(cfg) for cfg in itertools.product((0, 1), repeat=L_)]


def depth_of(cfg) -> int:
    """Highest active level (1-indexed); 0 if none on. Well-defined for any config.

    Under the Strict Supervenience Model the on-set is always {C1..Cm}, so
    depth = m fully describes the configuration. Under the generalised model a
    configuration may have gaps below its highest active level; depth is then a
    lossy summary and the per-level marginals P(C_i = 1 | e) are the primary
    quantity.
    """
    hi = 0
    for k in range(len(cfg)):
        if cfg[k]:
            hi = k + 1
    return hi


def _config_arrays(L_: int = L):
    """(configs list, config_matrix (n_cfg, L) int, depth_index array (n_cfg,))."""
    configs = _enumerate_configs_full(L_)
    cfg_mat = np.array(configs, dtype=int)            # (n_cfg, L)
    depth_idx = np.array([depth_of(c) for c in configs], dtype=int)
    return configs, cfg_mat, depth_idx


def _sample_edge_logprobs(prior_config: dict, n_draws: int, seed):
    """Per-draw log P(config | theta) for all 32 configs, bottom-up from C5.

    Returns (cfg_mat (n_cfg, L), depth_idx (n_cfg,), log_p_cfg (n_draws, n_cfg)).
    Draw order is fixed (pi, then alpha_C, then beta_C) so external code (e.g.
    the in-browser explorer) can reproduce it exactly.
    """
    configs, cfg_mat, depth_idx = _config_arrays(L)
    n_cfg = cfg_mat.shape[0]

    rng = np.random.default_rng(seed)
    pi = rng.beta(*prior_config["root"], size=n_draws)            # P(C5 = 1)
    if prior_config.get("strict", False):
        aC = np.ones((n_draws, L - 1))                            # Condition 2A
    else:
        aC = rng.beta(*prior_config["c_alpha"], size=(n_draws, L - 1))
    bC = rng.beta(*prior_config["c_beta"], size=(n_draws, L - 1))

    log_pi, log_1mpi = _safe_log(pi), _safe_log(1.0 - pi)
    log_aC, log_1maC = _safe_log(aC), _safe_log(1.0 - aC)
    log_bC, log_1mbC = _safe_log(bC), _safe_log(1.0 - bC)

    log_p_cfg = np.zeros((n_draws, n_cfg))
    for ci, cfg in enumerate(configs):
        lp = log_pi if cfg[L - 1] == 1 else log_1mpi             # root C5 (index L-1)
        # Edge e (0..L-2): parent = C_{e+2} (index e+1), child = C_{e+1} (index e).
        for e in range(L - 1):
            parent = cfg[e + 1]
            child = cfg[e]
            if parent == 1:
                lp = lp + (log_aC[:, e] if child == 1 else log_1maC[:, e])
            else:
                lp = lp + (log_bC[:, e] if child == 1 else log_1mbC[:, e])
        log_p_cfg[:, ci] = lp
    return cfg_mat, depth_idx, log_p_cfg


def prior_predictive_score(e_input,
                            prior_config: dict,
                            level_of: np.ndarray,
                            n_draws: int = 5000,
                            seed: int | None = 0,
                            return_per_theta: bool = False) -> dict:
    """Score one evidence vector e (indicator activations) under one prior config.

    Parameters
    ----------
    e_input : length-N_IND 0/1 array
        The observed evidence E = e: e_j = 1 if indicator I_j is judged present.
    prior_config : dict from `make_prior_config`
        Chooses the regime (strict / generalised) and all priors.
    level_of : length-N_IND int array
        0-indexed parent level of each indicator.

    Returns a dict with the per-level posteriors P(C_i = 1 | e), the posterior
    over depth (highest active level), and — with return_per_theta — the
    per-draw matrices needed for HDIs and for `overall_credence`.
    """
    pc = prior_config
    alpha = pc["indicator_alpha"]
    beta  = pc["indicator_beta"]
    N_IND = len(alpha)

    e = np.asarray(e_input, dtype=np.int64)
    if e.shape != (N_IND,):
        raise ValueError(f"e_input must have shape ({N_IND},); got {e.shape}")
    level_of = np.asarray(level_of, dtype=int)
    if level_of.shape != (N_IND,):
        raise ValueError(f"level_of must have shape ({N_IND},); got {level_of.shape}")

    # ---- Indicator messages (theta-independent — alpha/beta are fixed). ----
    m1 = np.where(e == 1, alpha, 1.0 - alpha)
    m0 = np.where(e == 1, beta,  1.0 - beta)
    log_m1 = _safe_log(m1)
    log_m0 = _safe_log(m0)

    log_top_0 = np.zeros(L)
    log_top_1 = np.zeros(L)
    for j in range(N_IND):
        k = int(level_of[j])
        log_top_0[k] += log_m0[j]
        log_top_1[k] += log_m1[j]

    # ---- Per-config log-prior (per draw) and log-likelihood (theta-independent). ----
    cfg_mat, depth_idx, log_p_cfg = _sample_edge_logprobs(pc, n_draws, seed)
    n_cfg = cfg_mat.shape[0]
    # log_lik_cfg[ci] = sum_i (c_i ? log_top_1[i] : log_top_0[i])
    log_lik_cfg = cfg_mat @ log_top_1 + (1 - cfg_mat) @ log_top_0   # (n_cfg,)

    # ---- Combine, normalise per draw, average over draws. ----
    log_joint = log_p_cfg + log_lik_cfg[None, :]
    log_joint -= log_joint.max(axis=1, keepdims=True)
    joint = np.exp(log_joint)
    joint /= joint.sum(axis=1, keepdims=True)            # P(cfg | e, theta), (n, n_cfg)

    # Per-level marginals: sum over configs with c_i = 1.
    post_C_per_theta = joint @ cfg_mat                   # (n, L)
    post_C = post_C_per_theta.mean(axis=0)

    # Depth distribution: group configs by highest active level.
    depth_mask = np.zeros((n_cfg, L + 1))
    depth_mask[np.arange(n_cfg), depth_idx] = 1.0
    post_dep_per_theta = joint @ depth_mask              # (n, L+1)
    post_dep = post_dep_per_theta.mean(axis=0)

    result = {
        "P(C_i=1|e)":        {i + 1: float(post_C[i]) for i in range(L)},
        "P(depth=d|e)":      {d:     float(post_dep[d]) for d in range(L + 1)},
        "most_likely_depth": int(np.argmax(post_dep)),
    }
    if return_per_theta:
        result["P(C_i=1|e,theta)_per_draw"]   = post_C_per_theta
        result["P(depth=d|e,theta)_per_draw"] = post_dep_per_theta
    return result


def overall_credence(post_C_per_theta, level_credences):
    """Overall credence that the system is conscious (paper Section 7.3).

    Implements  P(C | E) = sum_i P(L* = i) * P(C_i | E),  where L* is the
    critical level for consciousness and P(L* = i) are the theoretical
    credences — the degree of belief that Level i is the critical one, set by
    philosophical argument rather than by the system-level evidence.

    Parameters
    ----------
    post_C_per_theta : (n_draws, L) array
        Per-draw per-level posteriors P(C_i = 1 | e, theta), as returned by
        `prior_predictive_score(..., return_per_theta=True)`. A length-L vector
        of posterior means is also accepted.
    level_credences : length-L array
        Theoretical credences P(L* = i). Normalised to sum to 1 if they do not.

    Returns
    -------
    (n_draws,) array of per-draw overall credences (or a scalar if a length-L
    vector was passed), so HDIs can be computed downstream.
    """
    w = np.asarray(level_credences, dtype=float).ravel()
    if w.shape != (L,):
        raise ValueError(f"level_credences must have length {L}; got {w.shape}")
    if w.sum() <= 0:
        raise ValueError("level_credences must have positive sum")
    w = w / w.sum()
    post = np.asarray(post_C_per_theta, dtype=float)
    return post @ w


def prior_depth_distribution(prior_config: dict, n_draws: int = 5000, seed: int | None = 0) -> np.ndarray:
    """Prior over depth induced by the C-layer prior alone (no evidence). Length L+1."""
    cfg_mat, depth_idx, log_p_cfg = _sample_edge_logprobs(prior_config, n_draws, seed)
    n_cfg = cfg_mat.shape[0]
    p_cfg = np.exp(log_p_cfg)
    p_cfg /= p_cfg.sum(axis=1, keepdims=True)            # defensive renormalise
    depth_mask = np.zeros((n_cfg, L + 1))
    depth_mask[np.arange(n_cfg), depth_idx] = 1.0
    return (p_cfg @ depth_mask).mean(axis=0)


def prior_C_marginals(prior_config: dict, n_draws: int = 5000, seed: int | None = 0) -> np.ndarray:
    """Induced prior P(C_i = 1) for i = 1..L (length L). Computed directly from
    the full 32-config joint (valid in both regimes)."""
    cfg_mat, depth_idx, log_p_cfg = _sample_edge_logprobs(prior_config, n_draws, seed)
    p_cfg = np.exp(log_p_cfg)
    p_cfg /= p_cfg.sum(axis=1, keepdims=True)
    return (p_cfg @ cfg_mat).mean(axis=0)


def prior_C_from_depth(depth_prior) -> np.ndarray:
    """Marginal P(C_i = 1) implied by a depth distribution, ASSUMING staircases.

    P(C_i = 1) = P(depth >= i). Valid only under the Strict Supervenience Model
    (the on-set is downward closed). For the generalised model use
    `prior_C_marginals`.
    """
    dep = np.asarray(depth_prior, dtype=float).ravel()
    L_ = dep.shape[0] - 1
    out = np.zeros(L_)
    for k in range(L_):
        out[k] = dep[k + 1:].sum()
    return out


def format_score(scored: dict,
                 prior_depth: np.ndarray | None = None,
                 prior_C: np.ndarray | None = None) -> str:
    """Pretty-print a scored case."""
    lines = []
    lines.append("Per-level posteriors P(C_i = 1 | e):")
    for i, v in scored["P(C_i=1|e)"].items():
        bar = "#" * int(round(v * 40))
        suffix = f"    prior = {prior_C[i-1]:.3f}" if prior_C is not None else ""
        lines.append(f"  C{i}: {v:.3f}  {bar}{suffix}")
    lines.append("")
    lines.append("Posterior over depth (highest active level):")
    for d, v in scored["P(depth=d|e)"].items():
        bar = "#" * int(round(v * 40))
        suffix = f"    prior = {prior_depth[d]:.3f}" if prior_depth is not None else ""
        lines.append(f"  depth = {d}: {v:.3f}  {bar}{suffix}")
    lines.append("")
    lines.append(f"Most likely depth: {scored['most_likely_depth']}")
    return "\n".join(lines)
