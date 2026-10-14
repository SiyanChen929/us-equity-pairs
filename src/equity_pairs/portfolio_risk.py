"""Portfolio-risk helpers for additive pair-strategy experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


def cash_aware_inverse_volatility(
    returns: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    requested_gross: float,
    pair_cap: float,
    name_cap: float,
    sector_cap: float,
) -> tuple[pd.Series, dict[str, Any]]:
    """Fit inverse-volatility weights subject to pair, name, and sector caps."""

    assets = list(returns.columns.astype(str))
    count = len(assets)
    if not count or requested_gross <= 0:
        return pd.Series(0.0, index=assets), {
            "requested_gross": requested_gross,
            "effective_gross": 0.0,
            "cash_weight": 1.0,
            "optimization_success": True,
        }
    meta = metadata.set_index("pair").reindex(assets)
    if meta[["dependent", "independent", "sector"]].isna().any().any():
        raise RuntimeError("allocation metadata is incomplete")
    vol = returns.std(ddof=1).astype(float)
    positive = vol[vol > 1e-12]
    fallback = float(positive.median()) if len(positive) else 1.0
    vol = vol.where(vol > 1e-12, fallback)
    target = (1.0 / vol).to_numpy(dtype=float, copy=True)
    target = target / target.sum()

    constraints: list[np.ndarray] = []
    limits: list[float] = []
    names = sorted(
        set(meta["dependent"].astype(str)) | set(meta["independent"].astype(str))
    )
    for name in names:
        constraints.append(
            (
                meta["dependent"].astype(str).eq(name)
                | meta["independent"].astype(str).eq(name)
            )
            .astype(float)
            .to_numpy()
        )
        limits.append(float(name_cap))
    for sector in sorted(meta["sector"].astype(str).unique()):
        constraints.append(
            meta["sector"].astype(str).eq(sector).astype(float).to_numpy()
        )
        limits.append(float(sector_cap))
    matrix = np.vstack(constraints) if constraints else None
    bounds = [(0.0, float(pair_cap)) for _ in assets]
    capacity = linprog(
        -np.ones(count),
        A_ub=matrix,
        b_ub=np.asarray(limits) if limits else None,
        bounds=bounds,
        method="highs",
    )
    if not capacity.success:
        raise RuntimeError("allocation capacity problem is infeasible")
    effective_gross = min(float(requested_gross), float(np.sum(capacity.x)))
    scaled_target = target * effective_gross
    slsqp_constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - effective_gross)}
    ]
    if matrix is not None:
        for vector, limit in zip(matrix, limits, strict=True):
            slsqp_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, row=vector, cap=limit: float(
                        cap - row @ weights
                    ),
                }
            )
    result = minimize(
        lambda weights: float(np.sum(np.square(weights - scaled_target))),
        np.asarray(capacity.x)
        * (effective_gross / max(float(np.sum(capacity.x)), 1e-12)),
        method="SLSQP",
        bounds=bounds,
        constraints=slsqp_constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(
            "constrained inverse-volatility allocation failed: " + result.message
        )
    weights = pd.Series(result.x, index=assets, dtype=float)
    weights[weights.abs() < 1e-12] = 0.0
    return weights, {
        "requested_gross": float(requested_gross),
        "effective_gross": float(weights.sum()),
        "cash_weight": float(1.0 - weights.sum()),
        "optimization_success": True,
        "maximum_pair_weight": float(weights.max()),
    }
