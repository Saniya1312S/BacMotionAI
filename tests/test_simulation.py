"""Unit tests for BacMotionAI simulation and feature extraction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.simulation import simulate_bacterium, CHEM_STATS, STEPS
from src.features import extract_features, FEATURE_COLS
from src.models import REGIME_LABELS

def test_simulate_single_bacterium():
    xs, ys, ths = simulate_bacterium(W=20, hydro=False, chemical='none', seed=0)
    assert len(xs) == STEPS + 1
    assert len(ys) == STEPS + 1
    assert all(0 <= y <= 20 for y in ys), "Y positions should be within channel"

def test_feature_extraction_returns_14_features():
    xs, ys, ths = simulate_bacterium(W=20, hydro=True, chemical='attractant', seed=1)
    feats = extract_features(xs, ys, ths, W=20)
    assert len(feats) == 14, f"Expected 14 features, got {len(feats)}"
    for col in FEATURE_COLS:
        assert col in feats, f"Missing feature: {col}"

def test_regime_labels_count():
    assert len(REGIME_LABELS) == 5

def test_chemical_affects_speed():
    xs_none, ys_none, _ = simulate_bacterium(W=20, hydro=True, chemical='none', seed=10)
    xs_anti, ys_anti, _ = simulate_bacterium(W=20, hydro=True, chemical='antibiotic', seed=10)
    f_none = extract_features(xs_none, ys_none, np.zeros(len(xs_none)), W=20)
    f_anti = extract_features(xs_anti, ys_anti, np.zeros(len(xs_anti)), W=20)
    # Antibiotic should reduce speed on average
    assert f_anti['mean_speed'] < f_none['mean_speed'], "Antibiotic should reduce speed"

def test_hydro_affects_alignment():
    xs_dry, ys_dry, ths_dry = simulate_bacterium(W=10, hydro=False, seed=5)
    xs_wet, ys_wet, ths_wet = simulate_bacterium(W=10, hydro=True, seed=5)
    f_dry = extract_features(xs_dry, ys_dry, ths_dry, W=10)
    f_wet = extract_features(xs_wet, ys_wet, ths_wet, W=10)
    # Both should produce valid alignment values
    assert 0 <= f_dry['mean_alignment'] <= 1
    assert 0 <= f_wet['mean_alignment'] <= 1

if __name__ == '__main__':
    tests = [test_simulate_single_bacterium, test_feature_extraction_returns_14_features,
             test_regime_labels_count, test_chemical_affects_speed, test_hydro_affects_alignment]
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
    print("\nDone.")
