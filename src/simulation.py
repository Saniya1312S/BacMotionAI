"""
BacMotionAI — Physics-Based Bacterial Simulation Engine

Simulates E. coli run-and-tumble motion in confined micro-channels
with hydrodynamic wall interactions and chemical modulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ─── Physics Constants ───
V0 = 20.0        # Intrinsic swimming speed (µm/s)
DR = 0.2         # Rotational diffusion coefficient (rad²/s)
DT = 0.01        # Time step (s)
T_TOTAL = 6.0    # Total simulation time (s)
STEPS = int(T_TOTAL / DT)
L = 200.0        # Channel length (µm)
A = 1.0          # Cell body radius (µm)
KAPPA = 6.0      # Hydrodynamic torque strength
EPS = 0.12       # Near-wall speed enhancement factor

# ─── Chemical Parameter Distributions ───
CHEM_STATS = {
    'none':       {'v_mu': 1.00, 'v_sig': 0.05, 'Dr_mu': 1.0,  'Dr_sig': 0.1,  'drift_mu':  0.00, 'drift_sig': 0.05},
    'attractant': {'v_mu': 1.12, 'v_sig': 0.08, 'Dr_mu': 0.65, 'Dr_sig': 0.12, 'drift_mu':  0.28, 'drift_sig': 0.08},
    'repellent':  {'v_mu': 0.90, 'v_sig': 0.07, 'Dr_mu': 1.45, 'Dr_sig': 0.15, 'drift_mu': -0.22, 'drift_sig': 0.07},
    'antibiotic': {'v_mu': 0.52, 'v_sig': 0.10, 'Dr_mu': 2.40, 'Dr_sig': 0.30, 'drift_mu':  0.00, 'drift_sig': 0.04},
}

TUMBLE_RATES = {
    'none': 0.005,
    'attractant': 0.003,
    'repellent': 0.005,
    'antibiotic': 0.01,
}


def simulate_bacterium(
    W: float,
    hydro: bool = False,
    chemical: str = 'none',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a single bacterium trajectory in a confined channel.

    Parameters
    ----------
    W : float
        Channel width in µm (5–100).
    hydro : bool
        If True, enable hydrodynamic wall interactions.
    chemical : str
        Chemical condition: 'none', 'attractant', 'repellent', 'antibiotic'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xs, ys, thetas : np.ndarray
        Arrays of x-positions, y-positions, and orientations over time.
    """
    if seed is not None:
        np.random.seed(seed)

    cs = CHEM_STATS[chemical]
    v_scale = np.clip(np.random.normal(cs['v_mu'], cs['v_sig']), 0.3, 1.5)
    Dr_eff = np.clip(np.random.normal(cs['Dr_mu'], cs['Dr_sig']), 0.1, 5.0) * DR
    drift = np.random.normal(cs['drift_mu'], cs['drift_sig'])
    p_tumble = TUMBLE_RATES.get(chemical, 0.005)

    x = np.random.rand() * L
    y = np.random.rand() * W
    theta = np.random.rand() * 2 * np.pi

    xs, ys, thetas = [x], [y], [theta]

    for _ in range(STEPS):
        # Tumbling
        if np.random.rand() < p_tumble:
            theta = np.random.rand() * 2 * np.pi

        # Hydrodynamic wall interaction
        omega = 0.0
        v_eff = V0 * v_scale

        if hydro:
            h = min(y, W - y)
            h_eff = max(h, A)
            omega = -KAPPA * (A / h_eff) ** 2 * np.sin(2 * theta)
            v_eff = V0 * v_scale * (1 + EPS * (A / h_eff))

        # Update orientation with noise
        theta += omega * DT + drift * DT + np.sqrt(2 * Dr_eff * DT) * np.random.randn()

        # Update position
        x += v_eff * np.cos(theta) * DT
        y += v_eff * np.sin(theta) * DT

        # Periodic boundary (x-axis)
        x = x % L

        # Reflecting boundary (y-axis / walls)
        if y < 0:
            y = -y
            theta = -theta
        elif y > W:
            y = 2 * W - y
            theta = -theta

        xs.append(x)
        ys.append(y)
        thetas.append(theta)

    return np.array(xs), np.array(ys), np.array(thetas)


def run_batch_simulation(
    W: float,
    hydro: bool = False,
    chemical: str = 'none',
    N: int = 20,
    seed: int = 42
) -> List[Dict]:
    """
    Run simulation for N bacteria and return feature records.

    Parameters
    ----------
    W : float
        Channel width in µm.
    hydro : bool
        Enable hydrodynamic interactions.
    chemical : str
        Chemical condition.
    N : int
        Number of bacteria to simulate.
    seed : int
        Base random seed.

    Returns
    -------
    records : list of dict
        Each dict contains the 14 extracted features plus metadata.
    """
    from .features import extract_features

    records = []
    for i in range(N):
        xs, ys, thetas = simulate_bacterium(W, hydro, chemical, seed=seed + i)
        feats = extract_features(xs, ys, thetas, W, DT)
        feats['channel_width'] = W
        feats['hydro'] = hydro
        feats['chemical'] = chemical
        records.append(feats)

    return records
