"""
BacMotionAI — AI-Powered Bacterial Motility Analysis

Modules:
    simulation: Physics-based bacterial trajectory simulation
    features:   Feature extraction (14 features per trajectory)
    models:     ML/DL model configurations for regime classification
"""

from .simulation import simulate_bacterium, run_batch_simulation
from .features import extract_features, FEATURE_COLS
from .models import (
    build_random_forest,
    build_logistic_regression,
    build_xgboost,
    build_mlp,
    REGIME_LABELS,
)

__version__ = "2.0.0"
__author__ = "Saniya Saratkar"
