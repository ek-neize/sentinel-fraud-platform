"""Population Stability Index (PSI): has the input or score distribution shifted?

PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%). Bins come from the reference
period (quantiles for numeric features, levels for categoricals), so the reference
always scores 0. Conventional reading:

    < 0.10  stable        0.10 - 0.25  moderate shift        > 0.25  major shift

PSI needs no labels, so it is the first warning sign while chargebacks are still
arriving. It says *something changed*, not that the model got worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4
STABLE, MODERATE = 0.10, 0.25


def _psi(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    r = np.clip(ref_share, EPS, None)
    c = np.clip(cur_share, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def quantile_edges(ref, bins: int = 10) -> np.ndarray:
    """Interior cut points from the reference distribution (duplicates removed)."""
    vals = pd.Series(ref, dtype=float).dropna()
    if vals.empty:
        return np.array([])
    return np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)[1:-1]))


def _numeric_shares(values, edges: np.ndarray) -> np.ndarray:
    v = pd.Series(values, dtype=float)
    shares = []
    missing = v.isna()
    if len(v):
        b = np.searchsorted(edges, v[~missing].to_numpy(), side="right")
        counts = np.bincount(b, minlength=len(edges) + 1).astype(float)
        shares = np.r_[counts, missing.sum()] / len(v)
    else:
        shares = np.zeros(len(edges) + 2)
    return np.asarray(shares)


def psi_numeric(ref, cur, bins: int = 10) -> float:
    """PSI for a numeric column; missing values form their own bin."""
    edges = quantile_edges(ref, bins)
    return _psi(_numeric_shares(ref, edges), _numeric_shares(cur, edges))


def psi_categorical(ref, cur) -> float:
    r = pd.Series(ref).astype(str).value_counts(normalize=True)
    c = pd.Series(cur).astype(str).value_counts(normalize=True)
    levels = r.index.union(c.index)
    return _psi(
        r.reindex(levels, fill_value=0).to_numpy(), c.reindex(levels, fill_value=0).to_numpy()
    )


def drift_status(psi: float) -> str:
    if psi < STABLE:
        return "stable"
    if psi < MODERATE:
        return "moderate"
    return "major"


def drift_table(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    numeric: list[str],
    categorical: list[str] | None = None,
    period_col: str = "period",
    bins: int = 10,
) -> pd.DataFrame:
    """PSI per feature for every period in `cur` against the whole reference frame."""
    rows = []
    edges = {f: quantile_edges(ref[f], bins) for f in numeric}
    ref_shares = {f: _numeric_shares(ref[f], edges[f]) for f in numeric}
    for period, part in cur.groupby(period_col, sort=True):
        for f in numeric:
            rows.append((period, f, _psi(ref_shares[f], _numeric_shares(part[f], edges[f]))))
        for f in categorical or []:
            rows.append((period, f, psi_categorical(ref[f], part[f])))
    out = pd.DataFrame(rows, columns=[period_col, "feature", "psi"])
    out["status"] = out["psi"].map(drift_status)
    return out
