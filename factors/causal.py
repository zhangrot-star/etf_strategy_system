"""Causal inference framework using linearmodels for panel data.

Provides DID (Difference-in-Differences) and IV (Instrumental Variables)
interfaces to isolate policy-driven alpha from market beta contamination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, FirstDifferenceOLS, RandomEffects
from linearmodels.iv import IV2SLS

logger = logging.getLogger(__name__)


@dataclass
class CausalResult:
    """Container for causal inference results."""

    method: str
    coefficient: float
    std_error: float
    p_value: float
    r_squared: float
    n_observations: int
    summary: str = ""


class CausalInferenceEngine:
    """Panel data causal inference using linearmodels.

    Designed to answer:
    1. DID: What is the pure policy treatment effect after stripping market beta?
    2. IV:  What is the causal effect of fund flows on returns, instrumented by
           exogenous policy changes?
    """

    # ── Difference-in-Differences ────────────────────────────

    def run_did(
        self,
        data: pd.DataFrame,
        outcome_col: str = "return",
        treatment_col: str = "treated",
        post_col: str = "post",
        entity_col: str = "ticker",
        time_col: str = "trade_date",
        control_vars: list[str] | None = None,
    ) -> CausalResult:
        """Estimate treatment effect via canonical 2×2 DiD.

        Args:
            data: Panel DataFrame with entity, time, outcome, treatment, and post columns.
            outcome_col: Column name for the outcome variable (e.g., daily return).
            treatment_col: Binary indicator (1 = treated group).
            post_col: Binary indicator (1 = post-treatment period).
            entity_col: Entity identifier (e.g., ticker).
            time_col: Time identifier.
            control_vars: Additional covariates (e.g., market return, sector return)
                          to partial out confounding from market beta.

        Returns:
            CausalResult with the DID coefficient (policy treatment effect).
        """
        df = data.copy()
        df["_did_interact"] = df[treatment_col] * df[post_col]

        exog_vars = [treatment_col, post_col, "_did_interact"]
        if control_vars:
            exog_vars.extend(control_vars)

        # Clean data
        model_df = df[[entity_col, time_col, outcome_col] + exog_vars].dropna()
        model_df = model_df.set_index([entity_col, time_col])

        try:
            model = PanelOLS(
                model_df[outcome_col],
                model_df[exog_vars],
                entity_effects=True,
                time_effects=True,
            )
            results = model.fit(cov_type="clustered", cluster_entity=True)

            coef = results.params.get("_did_interact", 0.0)
            std_err = results.std_errors.get("_did_interact", np.nan)
            p_val = results.pvalues.get("_did_interact", np.nan)
            r2 = results.rsquared_inclusive

            logger.info(
                "DiD estimate: coef=%.6f, se=%.6f, p=%.4f, n=%d",
                coef, std_err, p_val, results.nobs,
            )

            return CausalResult(
                method="DID",
                coefficient=coef,
                std_error=std_err,
                p_value=p_val,
                r_squared=r2,
                n_observations=int(results.nobs),
                summary=(
                    f"Policy treatment effect: {coef:.4f} ({std_err:.4f}). "
                    f"{'Significant at 5%' if p_val < 0.05 else 'Not significant at 5%'}."
                ),
            )

        except Exception:
            logger.exception("DiD estimation failed")
            return CausalResult(
                method="DID",
                coefficient=np.nan,
                std_error=np.nan,
                p_value=np.nan,
                r_squared=np.nan,
                n_observations=len(model_df),
                summary="DiD estimation failed — check panel structure.",
            )

    # ── Instrumental Variables (2SLS) ─────────────────────────

    def run_iv(
        self,
        data: pd.DataFrame,
        outcome_col: str = "return",
        endogenous_col: str = "fund_flow",
        instrument_col: str = "policy_shock",
        entity_col: str = "ticker",
        time_col: str = "trade_date",
        control_vars: list[str] | None = None,
    ) -> CausalResult:
        """Two-stage least squares for instrumented causal inference.

        Typical use case:
        - endogenous_col: ETF fund flows (potentially confounded by returns)
        - instrument_col: exogenous policy rate shock (affects flows but not
          returns directly except through flows)

        Args:
            data: Panel DataFrame.
            outcome_col: Dependent variable column.
            endogenous_col: Endogenous regressor.
            instrument_col: Excluded instrument (must be correlated with
                            endogenous_col, uncorrelated with error term).
            entity_col: Entity identifier.
            time_col: Time identifier.
            control_vars: Additional exogenous controls.

        Returns:
            CausalResult with the IV coefficient.
        """
        df = data.copy()

        exog_vars: list[str] = []
        if control_vars:
            exog_vars.extend(control_vars)

        formula_parts = [f"{outcome_col} ~ 1 + [{endogenous_col} ~ {instrument_col}]"]
        if exog_vars:
            formula_parts[0] += f" + {' + '.join(exog_vars)}"

        model_df = df[
            [outcome_col, endogenous_col, instrument_col] + exog_vars
        ].dropna()

        try:
            model = IV2SLS.from_formula(
                formula_parts[0],
                model_df,
            )
            results = model.fit(cov_type="robust")

            coef = results.params.get(endogenous_col, 0.0)
            std_err = results.std_errors.get(endogenous_col, np.nan)
            p_val = results.pvalues.get(endogenous_col, np.nan)
            r2 = results.rsquared

            logger.info(
                "IV estimate (%s → %s): coef=%.6f, se=%.6f, p=%.4f",
                endogenous_col, outcome_col, coef, std_err, p_val,
            )

            # First-stage F-statistic check for weak instrument
            f_stat = results.first_stage.diagnostics.get("f_stat", np.nan) if hasattr(results, "first_stage") else np.nan

            return CausalResult(
                method="IV-2SLS",
                coefficient=coef,
                std_error=std_err,
                p_value=p_val,
                r_squared=r2,
                n_observations=len(model_df),
                summary=(
                    f"Causal effect of {endogenous_col} on {outcome_col}: {coef:.4f} ({std_err:.4f}). "
                    f"First-stage F: {f_stat:.2f}. "
                    f"{'Weak instrument warning' if f_stat < 10 else 'Instrument appears valid'}."
                ),
            )

        except Exception:
            logger.exception("IV estimation failed")
            return CausalResult(
                method="IV-2SLS",
                coefficient=np.nan,
                std_error=np.nan,
                p_value=np.nan,
                r_squared=np.nan,
                n_observations=len(model_df),
                summary="IV estimation failed — check instrument validity.",
            )

    # ── Panel regression for beta decomposition ───────────────

    def run_panel_beta(
        self,
        data: pd.DataFrame,
        return_col: str = "return",
        market_col: str = "market_return",
        entity_col: str = "ticker",
        time_col: str = "trade_date",
    ) -> pd.DataFrame:
        """Estimate entity-level CAPM betas using panel regression.

        Returns DataFrame with ticker, beta, alpha, and r_squared.
        """
        df = data[[entity_col, time_col, return_col, market_col]].dropna()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index([entity_col, time_col])

        try:
            model = PanelOLS(df[return_col], df[[market_col]], entity_effects=True)
            results = model.fit()

            # Extract entity fixed effects as alphas, market coefficient as beta
            effects = results.estimated_effects.groupby(level=0).first()
            betas: list[dict[str, Any]] = []
            for entity in data[entity_col].unique():
                entity_effect = (
                    float(effects.loc[entity, "estimated_effects"])
                    if entity in effects.index
                    else 0.0
                )
                betas.append({
                    "ticker": entity,
                    "beta": results.params.get(market_col, 1.0),
                    "alpha": entity_effect,
                    "r_squared": results.rsquared_inclusive,
                })

            return pd.DataFrame(betas)

        except Exception:
            logger.exception("Panel beta estimation failed")
            return pd.DataFrame()
