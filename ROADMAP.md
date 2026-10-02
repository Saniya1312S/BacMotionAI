# 🗺️ BacMotionAI — Development Roadmap

> **Version:** 2.0 (Anti-Overfitting Edition)  
> **Last Updated:** October 2026  
> **Status:** Phase 5 Complete ✅ | Phase 6 In Planning 🔮

---

## 📋 Overview

This roadmap outlines the development journey of BacMotionAI from initial concept to a production-ready, anti-overfitting ML pipeline with an interactive web dashboard.

---

## ✅ Phase 1: Physics Simulation Engine
**Status:** Complete | **Timeline:** Week 1-2

### Objectives
- [x] Implement run-and-tumble bacterial motion model
- [x] Add hydrodynamic wall interactions (torque coupling)
- [x] Support variable channel widths (5–100 µm)
- [x] Implement Dry (no hydro) vs Wet (hydrodynamic) conditions
- [x] Add wall reflection boundary conditions
- [x] Periodic boundary conditions along channel length
- [x] Biological noise via rotational diffusion

### Key Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| Intrinsic speed v₀ | 20.0 | µm/s |
| Rotational diffusion Dᵣ | 0.2 | rad²/s |
| Hydrodynamic torque κ | 6.0 | — |
| Cell radius a | 1.0 | µm |
| Wall speed enhancement ε | 0.12 | — |
| Channel length L | 200.0 | µm |
| Time step dt | 0.01 | s |
| Simulation duration | 6.0 | s |

### Deliverables
- `get_trajectory()` function for single-bacterium simulation
- `run_simulation()` function for batch simulation (N bacteria)
- Trajectory visualization (Obj 1): Dry vs Wet × Channel Width

---

## ✅ Phase 2: Chemical Stimuli Integration
**Status:** Complete | **Timeline:** Week 2-3

### Objectives
- [x] Define 4 chemical conditions: None, Attractant, Repellent, Antibiotic
- [x] Per-chemical parameter distributions (speed scale, rotational diffusion, drift, tumble rate)
- [x] Stochastic sampling from chemical parameter distributions
- [x] Chemical stimuli analysis (Obj 2): Speed distributions + alignment analysis

### Chemical Parameter Table
| Chemical | Speed (v_µ) | Rot. Diff (Dᵣ_µ) | Drift (µ) | Tumble Rate |
|----------|:-:|:-:|:-:|:-:|
| None | 1.00 ± 0.05 | 1.0 ± 0.1 | 0.00 ± 0.05 | 0.005 |
| Attractant | 1.12 ± 0.08 | 0.65 ± 0.12 | +0.28 ± 0.08 | 0.003 |
| Repellent | 0.90 ± 0.07 | 1.45 ± 0.15 | -0.22 ± 0.07 | 0.005 |
| Antibiotic | 0.52 ± 0.10 | 2.40 ± 0.30 | 0.00 ± 0.04 | 0.010 |

### Deliverables
- Chemical modulation in simulation engine
- Speed distribution plots per chemical
- Summary bar charts (mean speed, mean alignment)

---

## ✅ Phase 3: Feature Engineering & ML Pipeline
**Status:** Complete | **Timeline:** Week 3-5

### Objectives
- [x] Extract 14 experimentally-measurable features per bacterium
- [x] Define 5 motility regimes based on observed behavior
- [x] Generate 2,000 sample dataset across parameter space
- [x] Train 4 ML classifiers with proper data hygiene

### 14 Engineered Features
```
Speed:       mean_speed, std_speed, q25_speed, q75_speed, cv_speed, 
             speed_skewness, speed_kurtosis
Orientation: mean_alignment, std_alignment
Wall:        wall_rate, near_wall_frac
Motility:    mean_run_len
Geometry:    confinement_idx
Transport:   eff_diffusivity
```

### 5 Motility Regimes
1. **Inhibited** — Low speed, high CV, short runs
2. **Wall-Guided** — High wall accumulation + alignment
3. **Chemotactic** — Long runs, directed motion
4. **Confined** — Frequent wall collisions
5. **Free-Swimming** — Open exploration, moderate speed

### ML Models Implemented
1. **Random Forest** — max_depth=8, n_estimators=200, min_samples_leaf=5
2. **Logistic Regression** — C=0.5, max_iter=1000
3. **XGBoost** — max_depth=4, learning_rate=0.05, 500 rounds with early stopping
4. **MLP** — (64, 32) hidden layers, alpha=0.01 regularization, early stopping

### Deliverables
- Feature extraction pipeline
- Regime classification system
- 4 trained models with evaluation metrics
- Classification reports, confusion matrices, ROC curves

---

## ✅ Phase 4: Anti-Overfitting Verification
**Status:** Complete (v2.0) | **Timeline:** Week 5-6

### Objectives
- [x] Fix scaler data leakage (Critical Bug #1)
- [x] Fix SHAP values indexing for newer versions (Critical Bug #2)
- [x] Fix bogus speed calculation using trig identity (Critical Bug #3)
- [x] Fix flat speed distribution in Obj 2 (Bug #4)
- [x] Fix XGBoost deprecated parameters (Bug #5)
- [x] Remove dead code in interactive widget (Bug #6)
- [x] Fix tumble rate mismatch (Bug #7)
- [x] Remove unused optuna dependency (Bug #8)
- [x] Verify CV-test gap < 5% for all models
- [x] Verify SMOTE applied to train split only
- [x] Verify regime labels from behavior, not input params

### Bug Fix Severity Matrix
| Bug | Severity | Impact |
|-----|----------|--------|
| Scaler leakage | 🔴 Critical | Test metrics inflated by ~3-5% |
| SHAP indexing | 🔴 Critical | Crash with SHAP ≥ 0.42 |
| Speed calculation | 🔴 Critical | Displayed speeds are meaningless |
| Speed distribution | 🟡 Medium | Obj 2 histograms are flat/misleading |
| XGBoost params | 🟡 Medium | Deprecation warnings |
| Dead code | 🟡 Medium | UI widget lag |
| Tumble mismatch | 🟡 Medium | Inconsistent results |
| Unused optuna | 🟢 Minor | Unnecessary install time |

### Deliverables
- `fix_notebook.py` — Automated patch script
- `BacMotionAI_v2_AntiOverfit_Fixed.ipynb` — Cleaned notebook
- Learning curves showing train-val convergence
- Overfitting diagnostic charts

---

## ✅ Phase 5: Interactive Web Dashboard
**Status:** Complete | **Timeline:** Week 6-8

### Objectives
- [x] Premium glassmorphism UI design
- [x] Animated hero section with bacterial particle background
- [x] 4-stage pipeline flowchart with interactive highlighting
- [x] 4 snapshot cards detailing each pipeline stage
- [x] 8 Chart.js analysis visualizations
- [x] Real-time bacterial simulator (Canvas API)
- [x] Simulation controls (width, bacteria count, environment, chemical)
- [x] Live feature extraction display with animated bars
- [x] ML regime prediction with probability bars
- [x] 4 live simulation charts (trajectory, speed, y-density, radar)
- [x] Responsive design (mobile + desktop)
- [x] Smooth scroll animations (IntersectionObserver)

### Dashboard Architecture
```
index.html
├── Hero Section (Canvas particle animation)
├── Pipeline Section
│   ├── Flow Chart (4 clickable steps)
│   └── Snapshot Cards (4 detail cards)
├── Analysis Section
│   ├── Speed Distribution by Chemical
│   ├── Model Comparison (Acc, AUC, LogLoss)
│   ├── Feature Importances (RF)
│   ├── Regime Class Distribution (Doughnut)
│   ├── Learning Curves (RF + MLP)
│   ├── Confusion Matrix (Bubble)
│   ├── ROC Curves (One-vs-Rest)
│   └── Overfitting Check (CV vs Test)
├── Simulator Section
│   ├── Controls Panel (sliders, toggles, buttons)
│   ├── Live Canvas (trajectory rendering)
│   ├── Feature Extraction Display
│   ├── ML Prediction Display
│   └── Live Charts (trajectory, speed, y-density, radar)
└── Footer
```

### Deliverables
- `index.html` — Dashboard structure (485 lines)
- `style.css` — Styling (1086 lines)
- `app.js` — Engine + charts (1138 lines)

---

## 🔮 Phase 6: Future Enhancements (Planned)

### 6.1 — 3D Simulation (Q1 2027)
- [ ] Extend to 3D channel geometry
- [ ] Add Z-axis wall interactions
- [ ] 3D trajectory visualization with Three.js

### 6.2 — Real Microscopy Data Integration (Q2 2027)
- [ ] Import real bacterial tracking data (CSV/TrackMate)
- [ ] Compare simulated vs real feature distributions
- [ ] Transfer learning from simulated to real data
- [ ] Domain adaptation techniques

### 6.3 — Advanced ML Models (Q2 2027)
- [ ] LSTM/GRU for time-series trajectory classification
- [ ] Convolutional Neural Networks on trajectory images
- [ ] Transformer-based sequence models
- [ ] AutoML hyperparameter search (Optuna)

### 6.4 — Deployment & Scaling (Q3 2027)
- [ ] Docker containerization
- [ ] FastAPI backend for ML inference
- [ ] WebSocket real-time simulation streaming
- [ ] GPU-accelerated batch simulation (CuPy/JAX)

### 6.5 — Extended Biology (Q3–Q4 2027)
- [ ] Multi-species simulation (E. coli, B. subtilis, P. aeruginosa)
- [ ] Biofilm formation dynamics
- [ ] Quorum sensing chemical gradients
- [ ] Phototaxis and magnetotaxis models

### 6.6 — Publication & Community (Q4 2027)
- [ ] Write research paper (Biophysical Journal target)
- [ ] PyPI package release (`pip install bacmotionai`)
- [ ] Interactive documentation with Jupyter Book
- [ ] Community contribution guidelines

---

## 📊 Metrics Summary

| Metric | Value |
|--------|-------|
| Total lines of code | ~3,200+ |
| ML models trained | 4 |
| Features engineered | 14 |
| Motility regimes | 5 |
| Training samples | 2,000 |
| Chart visualizations | 12 (8 static + 4 live) |
| Anti-overfitting fixes | 8 |
| Best model accuracy | ~78% (Random Forest) |
| Best ROC-AUC | ~0.94 (Random Forest) |
| Max overfitting gap | < 3% ✅ |

---

<div align="center">

**🦠 BacMotionAI v2 — Research Roadmap**

*From physics simulation to ML classification to interactive web dashboard*

</div>
