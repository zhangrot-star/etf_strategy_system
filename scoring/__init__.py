"""ETF multi-factor scoring framework — adapted from '工匠之选' methodology.

Three-module structure (100-point scale):
  Module 1 (10%): Fund issuer quality
  Module 2 (40%): Index / strategy quality
  Module 3 (50%): Individual fund evaluation

ML predictions from the XGBoost ensemble modulate the final score.
"""

from scoring.etf_scorer import ETFScorer, FundScore
from scoring.modulation import MLScoreModulator

__all__ = ["ETFScorer", "FundScore", "MLScoreModulator"]
