"""Factor registry — centralized metadata and computation dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorMeta:
    """Metadata for a single factor."""

    name: str
    category: str  # momentum | volatility | liquidity | sentiment | causal | macro
    description: str = ""
    compute_fn: Callable[[pd.DataFrame], pd.Series] | None = None


class FactorRegistry:
    """Registry pattern for factor computation and retrieval.

    Usage:
        registry = FactorRegistry()
        registry.register("roc_5d", "momentum", compute_fn)
        registry.register("atr_21d", "volatility", compute_fn)
        factors_df = registry.compute_all(data)
    """

    def __init__(self) -> None:
        self._factors: dict[str, FactorMeta] = {}

    def register(
        self,
        name: str,
        category: str,
        compute_fn: Callable[[pd.DataFrame], pd.Series],
        description: str = "",
    ) -> None:
        """Register a factor computation function."""
        self._factors[name] = FactorMeta(
            name=name,
            category=category,
            description=description,
            compute_fn=compute_fn,
        )
        logger.debug("Registered factor: %s [%s]", name, category)

    def unregister(self, name: str) -> None:
        self._factors.pop(name, None)

    @property
    def factor_names(self) -> list[str]:
        return sorted(self._factors.keys())

    def by_category(self, category: str) -> list[str]:
        return [n for n, m in self._factors.items() if m.category == category]

    @property
    def categories(self) -> list[str]:
        return sorted({m.category for m in self._factors.values()})

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all registered factors on the input DataFrame.

        The input DataFrame should have OHLCV columns.
        Returns a DataFrame with all factor values, indexed like input.
        """
        results: dict[str, pd.Series] = {}
        for name, meta in self._factors.items():
            if meta.compute_fn is None:
                continue
            try:
                results[name] = meta.compute_fn(data)
            except Exception:
                logger.exception("Factor '%s' computation failed", name)
        return pd.DataFrame(results)

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarizing registered factors."""
        return pd.DataFrame(
            [
                {"name": m.name, "category": m.category, "description": m.description}
                for m in self._factors.values()
            ]
        )
