"""Load the DCM-derived per-indicator priors from the CSV in `dcm priors/`.

Each row in `Indicators Caco Nodes DCM priors.csv` is a single indicator I_j.
The Digital Consciousness Model (DCM; Shiller 2026) characterises each of its
nodes by two labels — **support** (how much more likely the feature is to be
present in a conscious system than an unconscious one) and **demandingness**
(how rarely the feature is present in systems that are not conscious). Each of
our indicators is mapped to its closest DCM analogue, and those labels are
converted into an (alpha, beta) probability pair by `dcm priors/mapping.py`
(see `dcm priors/mapping.md` for the conversion). We just read the resulting
values here.

Vocabulary:
    alpha_j = P(I_j = 1 | parent C_i = 1)  — true-positive rate for indicator j
    beta_j  = P(I_j = 1 | parent C_i = 0)  — false-positive rate for indicator j
where C_i is the consciousness node at the indicator's level (paper Section 7.2).
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Indicator:
    j: int               # 0-indexed position in the loaded list
    level: int           # 1..L
    level_name: str      # 'Behavioural', 'Computational functional', etc.
    name: str
    description: str
    chosen_node: str     # the DCM node name we mapped to, or '(default fallback)'
    support: str         # the DCM support label, or '' for fallback rows
    demandingness: str   # the DCM demandingness label, or '' for fallback rows
    alpha: float         # P(I=1 | parent C = 1)
    beta:  float         # P(I=1 | parent C = 0)


def _parse_level(level_str: str) -> tuple[int, str]:
    """Split '2 – Computational functional' into (2, 'Computational functional')."""
    s = str(level_str).strip()
    # Robust to either an en-dash or hyphen-minus separator.
    for sep in ("–", "-"):
        if sep in s:
            head, tail = s.split(sep, 1)
            return int(head.strip()), tail.strip()
    raise ValueError(f"could not parse level string {level_str!r}")


def load_indicators(csv_path: str | pathlib.Path) -> list[Indicator]:
    df = pd.read_csv(csv_path)
    out: list[Indicator] = []
    for j, row in df.iterrows():
        level, level_name = _parse_level(row["Level of C"])
        out.append(Indicator(
            j=j,
            level=level,
            level_name=level_name,
            name=str(row["Name"]).strip(),
            description=str(row["Description"]).strip(),
            chosen_node=str(row["chosen dcm node"]).strip(),
            support=(str(row.get("chosen node support", "")).strip()
                     if not pd.isna(row.get("chosen node support", "")) else ""),
            demandingness=(str(row.get("chosen node demandingness", "")).strip()
                           if not pd.isna(row.get("chosen node demandingness", "")) else ""),
            alpha=float(row["prior alpha"]),
            beta=float(row["prior beta"]),
        ))
    return out


def alpha_beta_arrays(indicators: list[Indicator]) -> tuple[np.ndarray, np.ndarray]:
    """Stack the per-indicator (alpha, beta) into two length-N arrays."""
    alpha = np.array([i.alpha for i in indicators], dtype=float)
    beta  = np.array([i.beta  for i in indicators], dtype=float)
    return alpha, beta


def indicators_by_level(indicators: list[Indicator]) -> dict[int, list[Indicator]]:
    grouped: dict[int, list[Indicator]] = {}
    for ind in indicators:
        grouped.setdefault(ind.level, []).append(ind)
    return grouped


def level_counts(indicators: list[Indicator]) -> dict[int, int]:
    return {lvl: len(ix) for lvl, ix in indicators_by_level(indicators).items()}


def level_of_array(indicators: list[Indicator]) -> np.ndarray:
    """Length-N int array of the (0-indexed) level for each indicator."""
    return np.array([i.level - 1 for i in indicators], dtype=int)


DEFAULT_CSV_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "dcm priors"
    / "Indicators Caco Nodes DCM priors.csv"
)
