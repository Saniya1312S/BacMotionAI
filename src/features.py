"""
BacMotionAI — Feature Extraction Module

Extracts 14 experimentally-measurable features from bacterial trajectories,
mimicking real microscopy tracking data analysis.
"""

import numpy as np
from typing import Dict, List

# Ordered list of all 14 feature column names
FEATURE_COLS = [
    'mean_speed', 'std_speed', 'q25_speed', 'q75_speed',
    'mean_alignment', 'std_alignment',
    'wall_rate', 'near_wall_frac',
    'mean_run_len', 'confinement_idx',
    'cv_speed', 'eff_diffusivity',
    'speed_skewness', 'speed_kurtosis',
]


def extract_features(
    xs: np.ndarray,
    ys: np.ndarray,
    thetas: np.ndarray,
    W: float,
    dt: float = 0.01
) -> Dict[str, float]:
    """
    Extract 14 features from a single bacterium trajectory.

    Parameters
    ----------
    xs : np.ndarray
        X-positions over time.
    ys : np.ndarray
        Y-positions over time.
    thetas : np.ndarray
        Orientation angles over time.
    W : float
        Channel width in µm.
    dt : float
        Time step in seconds.

    Returns
    -------
    features : dict
        Dictionary of 14 feature name-value pairs.
    """
    # Compute instantaneous speeds from displacements
    dx = np.diff(xs)
    dy = np.diff(ys)
    speeds = np.sqrt(dx**2 + dy**2) / dt

    # Alignment with walls: |cos(theta)|
    cos_thetas = np.abs(np.cos(thetas[1:]))

    # Wall proximity
    wall_layer = 0.15 * W
    near_wall = ((ys[1:] < wall_layer) | (ys[1:] > W - wall_layer))
    near_wall_frac = np.mean(near_wall)

    # Wall collisions (reflection events)
    wall_hits = np.sum((ys[1:] <= 0.01) | (ys[1:] >= W - 0.01))
    total_steps = len(speeds)

    # Run lengths (steps between tumble-like direction changes)
    angle_changes = np.abs(np.diff(thetas))
    tumble_threshold = np.pi / 2
    tumble_events = np.where(angle_changes > tumble_threshold)[0]
    if len(tumble_events) > 1:
        run_lengths = np.diff(tumble_events)
        mean_run_len = float(np.mean(run_lengths))
    else:
        mean_run_len = float(total_steps)

    # Speed statistics
    mean_speed = float(np.mean(speeds))
    std_speed = float(np.std(speeds))
    q25_speed = float(np.percentile(speeds, 25))
    q75_speed = float(np.percentile(speeds, 75))
    cv_speed = std_speed / mean_speed if mean_speed > 0 else 0.0

    # Higher-order speed statistics
    if std_speed > 0:
        z_scores = (speeds - mean_speed) / std_speed
        speed_skewness = float(np.mean(z_scores**3))
        speed_kurtosis = float(np.mean(z_scores**4) - 3)
    else:
        speed_skewness = 0.0
        speed_kurtosis = 0.0

    # Cell radius (constant)
    a = 1.0
    Dr_eff = 0.2  # approximate

    features = {
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'q25_speed': q25_speed,
        'q75_speed': q75_speed,
        'mean_alignment': float(np.mean(cos_thetas)),
        'std_alignment': float(np.std(cos_thetas)),
        'wall_rate': wall_hits / total_steps if total_steps > 0 else 0.0,
        'near_wall_frac': float(near_wall_frac),
        'mean_run_len': mean_run_len,
        'confinement_idx': a / W,
        'cv_speed': cv_speed,
        'eff_diffusivity': min(mean_speed**2 / (2 * Dr_eff), 5000.0) if Dr_eff > 0 else 0.0,
        'speed_skewness': speed_skewness,
        'speed_kurtosis': speed_kurtosis,
    }

    return features
