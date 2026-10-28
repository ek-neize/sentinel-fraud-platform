"""Reason codes: which inputs pushed each alert's score up, in plain words.

- XGBoost: exact per-feature contributions (TreeSHAP, `pred_contribs=True`).
- Logistic regression: coefficient x standardised input.

One-hot columns are summed back to their original feature (e.g. all `category_*` columns
-> `category`), so reasons talk about the transaction, not the encoding. These are
**model signals**, not verified facts: the digest, dashboard and assistant say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.models.feature_sets import CATEGORICAL, NUMERIC

_PREFIXES = ("num__", "log__", "z__", "cat__")
_CATEGORICAL_LIKE = list(CATEGORICAL) + ["hour_of_day", "day_of_week"]


def raw_feature(encoded_name: str) -> str:
    """'log__amt_24h' -> 'amt_24h'; 'cat__category_shopping_net' -> 'category';
    'log__missingindicator_secs_since_prev_txn' -> 'secs_since_prev_txn'."""
    name = encoded_name
    for p in _PREFIXES:
        if name.startswith(p):
            name = name[len(p) :]
            break
    if name.startswith("missingindicator_"):
        return name[len("missingindicator_") :]
    if encoded_name.startswith("cat__"):
        for c in sorted(_CATEGORICAL_LIKE, key=len, reverse=True):
            if name.startswith(c + "_"):
                return c
    return name


def contributions(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row contribution of each *raw* feature to the model's log-odds score."""
    pre, model = pipe[:-1], pipe[-1]
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out())
    if hasattr(model, "coef_"):
        dense = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        contrib = dense * model.coef_[0]
    else:
        import xgboost as xgb

        booster = model.get_booster()
        contrib = booster.predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]  # drop bias
    enc = pd.DataFrame(contrib, columns=names, index=X.index)
    return enc.T.groupby([raw_feature(n) for n in names]).sum().T


def _money(v) -> str:
    return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"


def describe(feature: str, row: pd.Series) -> str:
    """Human sentence for one feature of one transaction."""
    v = row.get(feature)
    if feature in ("amt_to_prior_mean", "amt_zscore", "card_prior_amt_mean"):
        ratio = row.get("amt_to_prior_mean")
        if pd.notna(ratio):
            return f"amount is {ratio:.1f}x this card's average"
        return "first transactions on this card (no spending history)"
    if feature in ("amount", "log_amount"):
        amt = row.get("amount")
        if amt is None or pd.isna(amt):
            amt = np.expm1(row.get("log_amount", 0) or 0)
        return f"amount {_money(amt)}"
    if feature in ("amt_to_prior_max", "card_prior_amt_max"):
        r = row.get("amt_to_prior_max")
        return (
            f"{r:.1f}x this card's previous largest purchase"
            if pd.notna(r)
            else "no previous purchases on this card"
        )
    if feature.startswith("amt_") and feature[4:] in ("1h", "24h", "7d"):
        w = feature[4:]
        return f"{_money(v)} spent on this card in the previous {w}"
    if feature.startswith("cnt_"):
        w = feature[4:]
        return f"{int(v)} earlier transaction(s) on this card in the previous {w}"
    if feature == "secs_since_prev_txn":
        if pd.isna(v):
            return "first transaction seen on this card"
        return (
            f"only {v / 60:.0f} min since the card's previous transaction"
            if v < 3600
            else f"{v / 3600:.1f} h since the card's previous transaction"
        )
    if feature in ("is_night", "hour_of_day"):
        h = row.get("hour_of_day")
        return f"transaction at {int(h):02d}:00 (UTC)" if pd.notna(h) else "time of day"
    if feature == "day_of_week":
        names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        return f"day of week: {names[int(v) - 1]}" if pd.notna(v) else "day of week"
    if feature == "category":
        return f"merchant category: {v}"
    if feature == "state":
        return f"customer state: {v}"
    if feature == "wl_card_hit":
        return "card is on the fraud hot-list" if v == 1 else "card not on hot-list"
    if feature == "wl_merchant_hit":
        return "merchant is on the risk list" if v == 1 else "merchant not on risk list"
    if feature == "wl_any_hit":
        return "watchlist hit" if v == 1 else "no watchlist hit"
    if feature == "home_merch_km":
        return f"merchant {v:,.0f} km from the customer's home"
    if feature == "card_prior_txn_count":
        return f"card has {int(v)} earlier transaction(s)"
    if feature == "card_prior_amt_std":
        return "spending pattern of this card"
    if feature == "log_city_pop":
        return f"customer city population ~{np.expm1(v):,.0f}"
    return f"{feature} = {v}"


# Several encoded inputs describe the same fact; keep only the first one in the reasons.
_SAME_FACT = {
    "amt_zscore": "amt_to_prior_mean",
    "card_prior_amt_mean": "amt_to_prior_mean",
    "log_amount": "amount",
    "is_night": "hour_of_day",
    "card_prior_amt_max": "amt_to_prior_max",
    "wl_any_hit": "wl_card_hit",
}


MIN_REASON_WEIGHT = 0.05  # log-odds; smaller pushes are noise, not a reason


def top_reasons(
    pipe, X: pd.DataFrame, n: int = 3, min_weight: float = MIN_REASON_WEIGHT
) -> pd.DataFrame:
    """For each row: up to n reasons that *raised* the score, strongest first.

    Returns columns reason_1..n, reason_features (list) and reason_weights (list).
    """
    c = contributions(pipe, X)
    out = []
    for idx, row in c.iterrows():
        seen, reasons, feats, weights = set(), [], [], []
        for feat, w in row.sort_values(ascending=False).items():
            if w <= min_weight or len(reasons) >= n:
                break
            fact = _SAME_FACT.get(feat, feat)
            if fact in seen:
                continue
            seen.add(fact)
            reasons.append(describe(feat, X.loc[idx]))
            feats.append(feat)
            weights.append(round(float(w), 4))
        out.append(
            {
                **{f"reason_{i + 1}": (reasons[i] if i < len(reasons) else None) for i in range(n)},
                "reason_features": feats,
                "reason_weights": weights,
            }
        )
    return pd.DataFrame(out, index=X.index)


MODEL_INPUTS = NUMERIC + CATEGORICAL
