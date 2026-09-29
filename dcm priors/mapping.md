# DCM (support, demandingness) → cacophony (α, β) priors

## What this is for

The cacophony hierarchy model needs a `(TPR, FPR)` pair per indicator. The
DCM tells us, for each node, two labels:

- **demandingness** — how prevalent the feature is *in non-conscious systems*,
- **support** — how much more prevalent it is *in conscious systems* than in
  non-conscious ones.

This document records how we convert those labels into the two probability
parameters used by the cacophony model:

- `β` = FPR = `P(J = 1 | C = 0)` (the rate at which the expert says "yes"
  when consciousness is absent — equivalently, the background prevalence of
  the feature),
- `α` = TPR = `P(J = 1 | C = 1)` (the rate at which the expert says "yes"
  when consciousness is present).

The mapping has two steps; each step depends on exactly one of the DCM labels.

## Step 1 — demandingness sets β

The DCM text anchors *neutral* demandingness at the point where the feature
is equally likely to be present or absent when the system is not conscious
(`β = 0.5`). Demanding features are rarer than that in non-conscious systems
(small β); undemanding features are more common (large β).

We use a symmetric anchor scale around 0.5:

| Demandingness               | β   |
|-----------------------------|----:|
| Overwhelmingly demanding    | 0.01 |
| Strongly demanding          | 0.05 |
| Moderately demanding        | 0.10 |
| Weakly demanding            | 0.30 |
| Neutral                     | 0.50 |
| Weakly undemanding          | 0.70 |
| Moderately undemanding      | 0.90 |
| Strongly undemanding        | 0.95 |
| Overwhelmingly undemanding  | 0.99 |

The geometry of the scale is intentional: it's compressed near 0.5 (so
"weakly" vs "neutral" stays a small step) and stretched toward the extremes
(so "overwhelmingly" really pushes against the bound). This was deliberately
chosen to be more informative than a softer mapping like `(0.02, 0.10, 0.25,
0.40, 0.50, 0.60, 0.75, 0.90, 0.98)`, which we considered first but discarded
because it left each indicator carrying too little signal — see the
"identifiability" section below.

## Step 2 — support sets α via a headroom fraction

The DCM text defines *support* as a likelihood ratio: how much more likely is
the feature in conscious vs non-conscious systems. The clean reading is
therefore `support = LR+ = α/β`, which would give `α = β · LR+`.

That clean reading runs into the bound `α ≤ 1`: with even modest β, the
multiplier needed for "strong" or "overwhelming" support drives α past 1. So
we re-express the rule as a *headroom fraction*: support specifies how much of
the gap between β and 1 the feature claims.

For positive support, with `s ∈ [0, 1)`:

> α = β + s · (1 − β)

For countersupport (rare but possible — present in the DCM as e.g. "weak
undermining"), with `c ∈ [0, 1)`:

> α = β · (1 − c)

This is equivalent to `α = β · LR+` when β is small (the regime support was
originally calibrated in), and saturates smoothly at α → 1 as β grows — so
the *effective* LR+ shrinks for high β, which is the same pattern the DCM's
own amplification table exhibits.

We use these anchors (only the weak/moderate/strong rows appear in the
cacophony-indicator → DCM-node mapping data; the other rows are defensive):

| Support level                | s    | α formula              |
|------------------------------|------|------------------------|
| Overwhelming support         | 0.98 | α = β + 0.98·(1 − β)   |
| Strong support               | 0.90 | α = β + 0.90·(1 − β)   |
| Moderate support             | 0.70 | α = β + 0.70·(1 − β)   |
| Weak support                 | 0.50 | α = β + 0.50·(1 − β)   |
| No support                   | 0.00 | α = β                  |
| Weak countersupport          | c=0.50 | α = 0.50 · β         |
| Moderate countersupport      | c=0.70 | α = 0.30 · β         |
| Strong countersupport        | c=0.90 | α = 0.10 · β         |
| Overwhelming countersupport  | c=0.98 | α = 0.02 · β         |

## Resulting 3 × 9 α grid

α (TPR), with weak/moderate/strong support on rows and the nine
demandingness levels on columns (β shown in the header for reference):

| Support \ β →      | 0.01 OD | 0.05 SD | 0.10 MD | 0.30 WD | 0.50 N | 0.70 WU | 0.90 MU | 0.95 SU | 0.99 OU |
|--------------------|--------:|--------:|--------:|--------:|-------:|--------:|--------:|--------:|--------:|
| Strong support     | 0.90    | 0.91    | 0.91    | 0.93    | 0.95   | 0.97    | 0.99    | 0.995   | 0.999   |
| Moderate support   | 0.70    | 0.72    | 0.73    | 0.79    | 0.85   | 0.91    | 0.97    | 0.985   | 0.997   |
| Weak support       | 0.51    | 0.53    | 0.55    | 0.65    | 0.75   | 0.85    | 0.95    | 0.975   | 0.995   |

Abbreviations: OD = overwhelmingly demanding … OU = overwhelmingly undemanding;
N = neutral.

## Sanity check against the DCM's own Table 3

The DCM publishes example post-amplification likelihood ratios for selected
combinations. Restricted to the rows that fall in our support range:

| Row                                  | DCM (LR+, LR−) | Mapping (α, β) → (LR+, LR−)    |
|--------------------------------------|----------------|--------------------------------|
| Strong / strongly demanding          | (6.7, 0.30)    | (0.91, 0.05) → (18.1, 0.10)    |
| Strong / strongly undemanding        | (1.1, 0.14)    | (0.995, 0.95) → (1.05, 0.10)   |
| Moderate / overwhelmingly demanding  | (16.5, 0.69)   | (0.70, 0.01) → (70, 0.31)      |
| Moderate / moderately demanding      | (2.4, 0.55)    | (0.73, 0.10) → (7.3, 0.30)     |
| Weak / overwhelmingly demanding      | (9.8, 0.80)    | (0.51, 0.01) → (51, 0.50)      |
| Weak / weakly demanding              | (1.2, 0.88)    | (0.65, 0.30) → (2.17, 0.50)    |

The mapping is materially *more* confident than the DCM's amplified table —
LR+ values are 2–8× larger on the "demanding" side. That is the deliberate
move: each indicator carries more signal, which is what we want for the
cacophony hierarchy to identify per-case latent depth from only 10
indicators per level.

## Why we chose the more polarised anchors

The earlier draft used softer values:

| version | demanding side (β at OD, SD, MD, WD) | support s (W, M, S) |
|---------|--------------------------------------|---------------------|
| soft    | 0.02, 0.10, 0.25, 0.40               | 0.20, 0.50, 0.80    |
| **current** | **0.01, 0.05, 0.10, 0.30**           | **0.50, 0.70, 0.90**    |

What this buys: the per-indicator signal-to-noise (the gap `α − β`) roughly
doubles on the typical "supportive + demanding" rows.

| Combination                          | soft (α, β, gap) | current (α, β, gap) |
|--------------------------------------|------------------|----------------------|
| Strong / strongly demanding          | 0.82, 0.10, 0.72 | 0.91, 0.05, **0.86** |
| Moderate / moderately demanding      | 0.63, 0.25, 0.38 | 0.73, 0.10, **0.63** |
| Weak / weakly demanding              | 0.52, 0.40, 0.12 | 0.65, 0.30, **0.35** |

In identifiability terms, more gap means:

- **Per-case latent posteriors `P(C_k = 1 | J)`** sharpen — each judgement
  contributes more bits of evidence per case. Latent recovery improves.
- **Population-level pi1 and alpha_C[k]** become better-identified through
  those per-case posteriors.
- **Individual `alpha_I[j], beta_I[j]` are slightly *less* identifiable** in
  the corner regions (α near 1, β near 0) because the likelihood is locally
  flat there. This is okay: the priors will already be centred on the
  mapping values, so the posterior doesn't need to move.

The undemanding rows ("moderately undemanding", "strongly undemanding"
parents — and the corresponding "biological similarity" pattern) are
intrinsically near-noise: the gap stays small (~0.05) regardless of support,
because a feature that fires 90% of the time in non-conscious systems can't
carry much consciousness-specific signal. That's a feature, not a bug.

## Edge cases

The script handles four edge cases identically: each falls back to
`(moderate support, moderately demanding)` → (α, β) = (0.73, 0.10):

1. **CSV row has chosen-node = NA** (7 rows).
2. **Chosen node is itself a Stance** (1 row — `Embodied Agency`). Stances
   carry no s/d in the DCM.
3. **Chosen-node name doesn't resolve in the DCM** (currently none, but the
   script tolerates it).
4. **Chosen node carries unrecognised s/d labels** (none in this data, but
   defensive — covers future DCM revisions adding new label categories).

For these the prior is moderately confident (α = 0.73, β = 0.10), which is
the same as a typical "moderate support / moderately demanding" indicator.
That's a sensible neutral position when we don't know.

## Important consequence for the existing PyMC prior

The cacophony 2.x model currently uses

    alpha_I ~ Beta(6, 2)    # mean 0.75
    beta_I  ~ Beta(2, 6)    # mean 0.25

Those defaults sit roughly on the soft-mapping centre, but they conflict with
the polarised mapping in two places:

- **Undemanding indicators** have truth `β ≈ 0.70 – 0.99`, well above the
  prior's mean of 0.25. The prior will fight the truth on those rows.
- **Strongly-demanding indicators** have truth `β ≈ 0.01 – 0.10`, also off
  the prior's centre but in the other direction.

The clean fix is to centre each indicator's Beta prior on its mapping value:

    alpha_I[j] ~ Beta(k * α_map[j], k * (1 - α_map[j]))
    beta_I[j]  ~ Beta(k * β_map[j], k * (1 - β_map[j]))

with `k` around 10–20 (k = 10 gives roughly the same prior strength the
current Beta(6, 2) has). Without this, the smoke-test "truth in 95% CI"
count will drop noticeably for the polarised mapping; with it, it should
hold.

## Files in this folder

- `mapping.py` — pure function `support_demand_to_alpha_beta(support, demandingness)`
  returning the prior `(α, β)`. The full anchor tables live here.
- `build_chosen_priors.py` — reads `Indicators Caco Nodes DCM.csv` and
  `full_dcm_anon.json`, applies the mapping, writes
  `Indicators Caco Nodes DCM priors.csv`.
- `Indicators Caco Nodes DCM priors.csv` — the 37-row prior table. Columns:
  the original 5 from the input CSV plus `chosen node support`,
  `chosen node demandingness`, `depth in tree`,
  `immediate parent support`, `immediate parent demandingness`,
  `prior alpha`, `prior beta`.

If you want chosen-node type or the containing-stances list back for reader
context, copy those columns over from
`Indicators Caco Nodes DCM extended.csv`.
