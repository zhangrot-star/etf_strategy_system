#!/usr/bin/env python3
"""Re-train XGBoost classifier + multi-horizon regressors on the full dataset."""
from __future__ import annotations

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_pipeline.db_manager import DatabaseManager
from data_pipeline.models import ETFPrice
from config.settings import Settings
from core.feature_utils import build_features_and_labels, build_features_from_prices
from core.ensemble import XGBoostEnsemble
from prediction.regressor import MultiHorizonRegressor
from sqlalchemy import select

db = DatabaseManager(Settings())
settings = Settings()

with db._engine.connect() as conn:
    result = conn.execute(select(ETFPrice.ticker).distinct())
    tickers = sorted([r[0] for r in result.fetchall()])

print(f"Training on {len(tickers)} tickers")

prices = db.load_prices(tickers, pd.Timestamp("2022-01-01"), pd.Timestamp("2026-06-01"))
print(f"Loaded {len(prices)} price rows")

# ── 1. Train classifier (63d forward, 3-class) ─────────────────

features_cls, labels = build_features_and_labels(prices, forward_window=63)
print(f"Classifier features: {features_cls.shape}, Labels: {len(labels)}")

label_counts = labels.value_counts().to_dict()
print(f"Labels: SELL={label_counts.get(0,0)}, HOLD={label_counts.get(1,0)}, BUY={label_counts.get(2,0)}")

ensemble = XGBoostEnsemble(settings)
ensemble.fit(features_cls, labels)

os.makedirs("models", exist_ok=True)
ensemble.save("models/xgboost_etf")
print("Classifier saved to models/xgboost_etf.{xgb,pkl}")

# ── 2. Train multi-horizon regressors ──────────────────────────

features_reg = build_features_from_prices(prices)
print(f"\nRegressor features: {features_reg.shape}")

regressor = MultiHorizonRegressor(horizons=[5, 21, 63], settings=settings)
regressor.fit_all(features_reg, prices)
regressor.save_all("models/xgboost_reg")

print("Regressors saved:")
for h in [5, 21, 63]:
    p = f"models/xgboost_reg_{h}d"
    if os.path.exists(f"{p}.xgb"):
        print(f"  {p}.{{xgb,pkl}}")

# ── 3. Feature importance ──────────────────────────────────────

imp = ensemble.get_feature_importance()
print("\nTop-10 features (classifier):")
for feat, score in sorted(imp.items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat}: {score:.4f}")
