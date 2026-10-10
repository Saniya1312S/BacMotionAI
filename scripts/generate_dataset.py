"""
Generate the bacterial motility dataset (2000 samples).
Run: python scripts/generate_dataset.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.simulation import simulate_bacterium, CHEM_STATS, TUMBLE_RATES, DT
from src.features import extract_features, FEATURE_COLS

REGIME_LABELS = ['Inhibited', 'Wall-Guided', 'Chemotactic', 'Confined', 'Free-Swimming']

def assign_regime(f):
    """Assign regime label based on measured behavior (not input params)."""
    scores = {r: 0.0 for r in REGIME_LABELS}
    scores['Inhibited'] += max(0, (12 - f['mean_speed']) / 8) + min(f['cv_speed'], 1.0) * 0.5
    scores['Wall-Guided'] += f['near_wall_frac'] * 2.0 + (max(0, f['mean_alignment'] - 0.6) * 2.0)
    scores['Chemotactic'] += min(f['mean_run_len'] / 200, 1.5) + max(0, 0.5 - f['cv_speed']) * 1.5
    scores['Confined'] += f['wall_rate'] * 3.0 + max(0, (100 - f['mean_run_len']) / 100)
    scores['Free-Swimming'] += max(0, (f['mean_speed'] - 17) / 5) + max(0, 0.4 - f['near_wall_frac'])
    return max(scores, key=scores.get)

def main():
    widths = [5, 10, 15, 20, 30, 50, 75, 100]
    chemicals = list(CHEM_STATS.keys())
    hydro_opts = [False, True]
    records = []
    target = 2000
    seed = 0

    print(f"Generating {target} samples...")
    while len(records) < target:
        for W in widths:
            for chem in chemicals:
                for hydro in hydro_opts:
                    if len(records) >= target:
                        break
                    xs, ys, ths = simulate_bacterium(W, hydro, chem, seed=seed)
                    feats = extract_features(xs, ys, ths, W, DT)
                    feats['regime'] = assign_regime(feats)
                    feats['channel_width'] = W
                    feats['hydro'] = hydro
                    feats['chemical'] = chem
                    records.append(feats)
                    seed += 1
        if len(records) < target:
            print(f"  Generated {len(records)} so far...")

    df = pd.DataFrame(records[:target])
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), '..', 'data', 'bacterial_motility_dataset.csv')
    df.to_csv(out, index=False)
    print(f"\n✅ Saved {len(df)} samples to {out}")
    print(f"Regime distribution:\n{df['regime'].value_counts().to_string()}")

if __name__ == '__main__':
    main()
