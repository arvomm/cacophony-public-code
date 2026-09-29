"""DCM (support, demandingness) → cacophony (alpha, beta) prior mapping.

Two-step mapping:
  1. demandingness alone sets beta (= FPR)
  2. support alone sets s, then alpha = beta + s * (1 - beta)   for positive support
                       or  alpha = beta * (1 - c)               for countersupport

The anchors below are the "more informative" / "polarised" version (β pushed
toward 0 for demanding, toward 1 for undemanding; s pushed toward 1 for
strong). See mapping.md for derivation and trade-offs.
"""
from __future__ import annotations


BETA_BY_DEMANDINGNESS: dict[str, float] = {
    "overwhelmingly demanding":   0.01,
    "strongly demanding":         0.05,
    "moderately demanding":       0.10,
    "weakly demanding":           0.30,
    "neutral":                    0.50,
    "weakly undemanding":         0.70,
    "moderately undemanding":     0.90,
    "strongly undemanding":       0.95,
    "overwhelmingly undemanding": 0.99,
}

# Positive-support headroom fractions: alpha = beta + s * (1 - beta).
S_BY_SUPPORT: dict[str, float] = {
    "overwhelming support": 0.98,   # not seen in the chosen-node data; defensive
    "strong support":       0.90,
    "moderate support":     0.70,
    "weak support":         0.50,
    "no support":           0.00,
}

# Countersupport fractions: alpha = beta * (1 - c).
# "Undermining" treated as a synonym for countersupport (the DCM uses it
# interchangeably in a few places, e.g. "weak undermining").
C_BY_COUNTERSUPPORT: dict[str, float] = {
    "weak countersupport":         0.50,
    "moderate countersupport":     0.70,
    "strong countersupport":       0.90,
    "overwhelming countersupport": 0.98,
    "weak undermining":            0.50,
    "moderate undermining":        0.70,
    "strong undermining":          0.90,
    "overwhelming undermining":    0.98,
}

# Fallback for missing / NA / unrecognised labels (e.g. when the chosen node
# is a Stance, which carries no support or demandingness in the DCM).
FALLBACK_SUPPORT = "moderate support"
FALLBACK_DEMAND  = "moderately demanding"


def support_demand_to_alpha_beta(
    support: str | None,
    demandingness: str | None,
) -> tuple[float, float]:
    """Return (alpha, beta) for a single indicator given DCM-style labels.

    Unrecognised / missing labels fall back to (moderate support, moderately
    demanding) — α = 0.73, β = 0.10.

    Examples
    --------
    >>> support_demand_to_alpha_beta("strong support", "strongly demanding")
    (0.9050000000000001, 0.05)
    >>> support_demand_to_alpha_beta("moderate support", "moderately demanding")
    (0.73, 0.1)
    >>> support_demand_to_alpha_beta(None, "NA")
    (0.73, 0.1)
    """
    s_label = (support or "").strip().lower()
    d_label = (demandingness or "").strip().lower()

    # 1. beta from demandingness (fallback if missing/unknown)
    if d_label not in BETA_BY_DEMANDINGNESS:
        d_label = FALLBACK_DEMAND
    beta = BETA_BY_DEMANDINGNESS[d_label]

    # 2. alpha from support
    if s_label in S_BY_SUPPORT:
        s = S_BY_SUPPORT[s_label]
        alpha = beta + s * (1.0 - beta)
    elif s_label in C_BY_COUNTERSUPPORT:
        c = C_BY_COUNTERSUPPORT[s_label]
        alpha = beta * (1.0 - c)
    else:
        # missing/unknown: fallback to moderate support
        s = S_BY_SUPPORT[FALLBACK_SUPPORT]
        alpha = beta + s * (1.0 - beta)

    return alpha, beta


if __name__ == "__main__":
    # Print the full 3 x 9 alpha grid as a quick sanity check.
    print("alpha (TPR) grid — rows = support, cols = demandingness\n")
    cols = list(BETA_BY_DEMANDINGNESS)
    head = f"{'support \\ β':22s} " + " ".join(f"{BETA_BY_DEMANDINGNESS[d]:>6.2f}" for d in cols)
    print(head)
    for s_lbl in ("strong support", "moderate support", "weak support"):
        row = f"{s_lbl:22s} "
        for d_lbl in cols:
            a, _ = support_demand_to_alpha_beta(s_lbl, d_lbl)
            row += f" {a:>5.3f}"
        print(row)
