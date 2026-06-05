"""Tests for PredictionPipeline."""

from __future__ import annotations

import pandas as pd

from prediction.pipeline import PredictionPipeline


class TestPredictionPipeline:
    def test_run_empty_prices_returns_empty(self):
        pipeline = PredictionPipeline()
        result = pipeline.run(pd.DataFrame())
        assert result == []

    def test_run_unfitted_returns_empty(self, sample_prices_df):
        pipeline = PredictionPipeline()
        # No models trained, so is_fitted is False
        result = pipeline.run(sample_prices_df)
        assert result == []
