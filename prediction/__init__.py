"""ETF multi-horizon return prediction package."""

from prediction.regressor import (
    HorizonPrediction,
    MultiHorizonPrediction,
    MultiHorizonRegressor,
    XGBoostReturnRegressor,
    _compute_forward_returns,
)
from prediction.pipeline import PredictionPipeline
from prediction.schemas import ETFReturnPrediction, PredictionRunResult, PredictionSummary
from prediction.evaluator import PredictionEvaluator, EvaluationReport, HorizonMetrics

__all__ = [
    "HorizonPrediction",
    "MultiHorizonPrediction",
    "MultiHorizonRegressor",
    "XGBoostReturnRegressor",
    "_compute_forward_returns",
    "PredictionPipeline",
    "ETFReturnPrediction",
    "PredictionRunResult",
    "PredictionSummary",
    "PredictionEvaluator",
    "EvaluationReport",
    "HorizonMetrics",
]
