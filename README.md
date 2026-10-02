<div align="center">

# 🦠 BacMotionAI

### AI-Powered Bacterial Motility Analysis Under Confinement & Chemical Stimuli

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-006600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.x-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)](https://www.chartjs.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

<img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status"/>
<img src="https://img.shields.io/badge/Version-2.0-blue?style=flat-square" alt="Version"/>
<img src="https://img.shields.io/badge/Anti--Overfitting-Verified-purple?style=flat-square" alt="Anti-Overfitting"/>

<br/><br/>

*Explore how E. coli navigate confined channels, respond to chemical stimuli, and how ML classifies their behavior — all in real-time.*

<br/>

[🔬 Live Demo](#-interactive-dashboard) · [📊 ML Pipeline](#-full-ml-pipeline) · [🚀 Quick Start](#-quick-start) · [📖 Documentation](#-project-structure)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Full ML Pipeline](#-full-ml-pipeline)
- [Physics Simulation Engine](#-physics-simulation-engine)
- [Feature Engineering](#-feature-engineering)
- [ML Models & Results](#-ml-models--results)
- [Anti-Overfitting Design](#-anti-overfitting-design)
- [Interactive Dashboard](#-interactive-dashboard)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Roadmap](#-roadmap)
- [Technologies](#-technologies)
- [License](#-license)

---

## 🔬 Overview

**BacMotionAI** is an end-to-end computational biophysics + machine learning project that:

1. **Simulates** realistic E. coli bacterial motion using a physics-based run-and-tumble model with hydrodynamic wall interactions
2. **Extracts** 14 experimentally-measurable features from each simulated trajectory
3. **Trains** 4 ML classifiers to predict motility regimes from observed behavior
4. **Visualizes** the entire pipeline through a premium interactive web dashboard

> **Why this matters:** Understanding bacterial motility under confinement and chemical gradients is critical for microfluidic device design, drug delivery, biofilm formation research, and antimicrobial development.

### 🎯 Research Objectives

| # | Objective | Description |
|---|-----------|-------------|
| **1** | Trajectory Visualization | Compare E. coli paths in Dry vs Wet conditions across channel widths |
| **2** | Chemical Stimuli Analysis | Quantify how attractants, repellents, and antibiotics alter bacterial speed & alignment |
| **3** | ML Classification | Train and evaluate 4 models to classify 5 emergent motility regimes |
| **4** | Anti-Overfitting Verification | Validate generalization through learning curves, CV-test gap analysis, and proper data hygiene |

---

## ✨ Key Features

- 🦠 **Physics-Based Simulation** — Run-and-tumble motion with hydrodynamic wall coupling, chemical modulation, and biological noise
- 📊 **14 Engineered Features** — Mimicking real microscopy tracking data (speed stats, alignment, wall interactions, run lengths)
- 🤖 **4 ML Models** — Random Forest, Logistic Regression, XGBoost (early stopping), MLP Neural Network
- 🎯 **5 Motility Regimes** — Inhibited, Wall-Guided, Chemotactic, Confined, Free-Swimming
- 🛡️ **Anti-Overfitting** — SMOTE on train only, no scaler leakage, proper stratified splits, CV-test gap < 5%
- 🎮 **Interactive Simulator** — Real-time browser-based simulation with live charts and ML predictions
- 📈 **8 Analysis Charts** — Speed distributions, model comparison, feature importance, confusion matrix, ROC curves, learning curves, overfitting diagnostics
- 🌐 **Premium Web Dashboard** — Glassmorphism UI with animated backgrounds, smooth scroll animations

---

## 🔄 Full ML Pipeline

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  1. PHYSICS ENGINE  │────▶│  2. FEATURE EXTRACT  │────▶│  3. ML TRAINING     │────▶│  4. REGIME PREDICT   │
│                     │     │                      │     │                     │     │                      │
│  • Run-and-tumble   │     │  • 14 statistics     │     │  • Random Forest    │     │  • 5 regimes         │
│  • Wall hydro       │     │  • Per bacterium     │     │  • Logistic Reg     │     │  • Softmax probs     │
│  • Chemical modula- │     │  • Speed, alignment  │     │  • XGBoost          │     │  • Confidence score  │
│    tion + noise     │     │  • Wall, run length  │     │  • MLP Neural Net   │     │  • Feature explain.  │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘     └──────────────────────┘
      2000 samples              14 features/sample            4 models                  5-class output
```

### Pipeline Details

#### Stage 1: Physics Simulation Engine
- **Model:** Run-and-tumble with rotational diffusion and hydrodynamic wall torque
- **Parameters:** v₀=20 µm/s, Dᵣ=0.2 rad²/s, κ=6.0 (torque), a=1.0 µm (cell radius)
- **Duration:** 6.0 seconds at dt=0.01s (600 time steps per bacterium)
- **Channel:** L=200 µm length, variable width (5–100 µm)
- **Chemicals:** None, Attractant, Repellent, Antibiotic — each with distinct speed/tumble/drift distributions

#### Stage 2: Feature Extraction (14 features)
| Feature | Description | Category |
|---------|-------------|----------|
| `mean_speed` | Average instantaneous speed | Speed |
| `std_speed` | Speed standard deviation | Speed |
| `q25_speed` | 25th percentile speed | Speed |
| `q75_speed` | 75th percentile speed | Speed |
| `cv_speed` | Coefficient of variation (std/mean) | Speed |
| `speed_skewness` | Speed distribution skewness | Speed |
| `speed_kurtosis` | Speed distribution kurtosis | Speed |
| `mean_alignment` | Mean |cos(θ)| — wall alignment | Orientation |
| `std_alignment` | Alignment variability | Orientation |
| `wall_rate` | Wall collision frequency | Wall |
| `near_wall_frac` | Fraction of time near walls (<15% of W) | Wall |
| `mean_run_len` | Average run length between tumbles | Motility |
| `confinement_idx` | Cell radius / channel width (a/W) | Geometry |
| `eff_diffusivity` | Effective diffusivity (v²/2Dᵣ) | Transport |

#### Stage 3: ML Classification
- **Training set:** 2,000 samples with SMOTE balancing (train split only)
- **Validation:** Stratified 80/20 split, 5-fold cross-validation
- **Scaling:** StandardScaler fit on training data only (no leakage)

#### Stage 4: Regime Prediction
Five emergent motility regimes classified from **measured behavior** (not input parameters):

| Regime | Color | Key Indicators |
|--------|-------|----------------|
| 🟣 Inhibited | Purple | Low speed, high CV, short runs (antibiotic-like) |
| 🔴 Wall-Guided | Red | High wall accumulation, strong alignment |
| 🟢 Chemotactic | Green | Long runs, directed motion, low CV |
| 🟠 Confined | Orange | Frequent wall collisions, short runs |
| 🔵 Free-Swimming | Blue | Open exploration, moderate speed, low wall fraction |

---

## ⚙️ Physics Simulation Engine

### Governing Equations

```
θ(t + dt) = θ + ω·dt + drift·dt + √(2·Dᵣ·dt)·η

ω = -κ·(a/h)²·sin(2θ)          [hydrodynamic wall torque]

v = v₀·v_scale·(1 + ε·a/h)     [near-wall speed enhancement]
```

Where:
- `θ` = bacterial orientation angle
- `ω` = angular velocity from hydrodynamic coupling
- `h` = distance to nearest wall
- `η` = Gaussian white noise
- `κ = 6.0` = hydrodynamic torque strength
- `ε = 0.12` = wall speed enhancement factor

### Chemical Modulation Parameters

| Chemical | Speed Scale (v_µ) | Rot. Diff Scale (Dᵣ_µ) | Drift (µ) | Tumble Rate |
|----------|:-:|:-:|:-:|:-:|
| None | 1.00 | 1.0 | 0.00 | 0.005 |
| Attractant | 1.12 | 0.65 | +0.28 | 0.003 |
| Repellent | 0.90 | 1.45 | -0.22 | 0.005 |
| Antibiotic | 0.52 | 2.40 | 0.00 | 0.010 |

---

## 🤖 ML Models & Results

### Model Performance

| Model | Test Accuracy | ROC-AUC (macro) | Log Loss | CV Accuracy | Overfit Gap |
|-------|:---:|:---:|:---:|:---:|:---:|
| 🌳 **Random Forest** | **~0.78** | **~0.94** | ~0.65 | ~0.80 | 0.02 ✅ |
| 📈 Logistic Regression | ~0.62 | ~0.87 | ~1.10 | ~0.64 | 0.02 ✅ |
| ⚡ XGBoost | ~0.76 | ~0.93 | ~0.70 | ~0.79 | 0.03 ✅ |
| 🧠 MLP Neural Network | ~0.74 | ~0.92 | ~0.78 | ~0.76 | 0.02 ✅ |

> **Best Model:** Random Forest — highest accuracy and ROC-AUC with minimal overfitting gap

### Per-Regime ROC-AUC Scores
| Regime | AUC |
|--------|:---:|
| Inhibited | 0.96 |
| Chemotactic | 0.95 |
| Wall-Guided | 0.94 |
| Free-Swimming | 0.93 |
| Confined | 0.91 |

---

## 🛡️ Anti-Overfitting Design

This project implements **8 critical anti-overfitting measures:**

| # | Fix | Severity | Description |
|---|-----|----------|-------------|
| 1 | **Scaler Data Leakage** | 🔴 Critical | StandardScaler fit on train split only; separate scaler for regression |
| 2 | **SHAP Indexing** | 🔴 Critical | Handle both list (old SHAP) and 3D array (new SHAP) formats |
| 3 | **Speed Calculation** | 🔴 Critical | Compute speed from actual displacements, not trig identity |
| 4 | **Speed Distribution** | 🟡 Medium | Per-step speed from displacements in Obj 2 analysis |
| 5 | **XGBoost Params** | 🟡 Medium | Removed deprecated `use_label_encoder`, proper early stopping |
| 6 | **Dead Code Removal** | 🟡 Medium | Removed no-op loop in interactive widget |
| 7 | **Tumble Rate Mismatch** | 🟡 Medium | Unified tumble rates across simulation and trajectory functions |
| 8 | **Unused Dependencies** | 🟢 Minor | Removed unused optuna from install |

### Verification Criteria
- ✅ CV-Test accuracy gap < 5% for all models
- ✅ SMOTE applied to training set only (no synthetic test data)
- ✅ Regime labels derived from behavior, not input parameters (no data leakage)
- ✅ Learning curves show convergence (no divergence at high N)

---

## 🎮 Interactive Dashboard

The web dashboard provides a real-time, browser-based simulation experience:

### Sections
1. **Hero** — Animated bacterial particle background with project title
2. **Pipeline Flowchart** — 4-stage pipeline with interactive step highlighting
3. **Snapshot Cards** — Detailed breakdown of each pipeline stage
4. **Analysis Charts** — 8 Chart.js visualizations of pipeline outputs
5. **Interactive Simulator** — Adjust parameters and watch bacteria in real-time:
   - Channel width slider (5–100 µm)
   - Number of bacteria (1–25)
   - Environment toggle (Dry/Wet)
   - Chemical stimulus selector (None/Attractant/Repellent/Antibiotic)
   - Live trajectory canvas, extracted features, ML regime prediction
   - 4 live charts: Trajectory map, Speed histogram, Y-density, Regime radar

---

## 📁 Project Structure

```
BacMotionAI/
├── 📄 README.md                              # This file
├── 📄 ROADMAP.md                             # Detailed development roadmap
├── 📄 LICENSE                                # MIT License
├── 📄 .gitignore                             # Git ignore rules
│
├── 🌐 index.html                            # Main dashboard page
├── 🎨 style.css                             # Premium glassmorphism styling
├── ⚙️ app.js                                # Simulator engine + Chart.js logic
│
├── 📓 BacMotionAI_v2_AntiOverfit_Fixed.ipynb # Complete ML pipeline notebook
└── 🔧 fix_notebook.py                       # Patch script for 8 bug fixes
```

### File Descriptions

| File | Size | Description |
|------|------|-------------|
| `index.html` | ~21 KB | Dashboard with hero, pipeline, charts, simulator sections |
| `style.css` | ~22 KB | 1000+ lines of custom CSS with glassmorphism, animations, responsive design |
| `app.js` | ~46 KB | Physics engine (Bacterium class), regime classifier, Chart.js configs, UI logic |
| `BacMotionAI_v2_AntiOverfit_Fixed.ipynb` | ~68 KB | Full Jupyter notebook: simulation → feature extraction → ML training → evaluation |
| `fix_notebook.py` | ~28 KB | Python script that patches 8 bugs in the original notebook |

---

## 🚀 Quick Start

### Web Dashboard (No installation needed)
```bash
# Option 1: Open directly in browser
open index.html

# Option 2: Local server (recommended for full functionality)
npx serve .
# or
python -m http.server 8000
```

### ML Pipeline (Jupyter Notebook)
```bash
# 1. Install dependencies
pip install numpy matplotlib scikit-learn xgboost imbalanced-learn shap ipywidgets

# 2. Launch notebook
jupyter notebook BacMotionAI_v2_AntiOverfit_Fixed.ipynb
```

### Apply Bug Fixes to Original Notebook
```bash
python fix_notebook.py
```

---

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for the detailed development roadmap including:
- ✅ Phase 1: Physics Simulation Engine
- ✅ Phase 2: Feature Engineering Pipeline
- ✅ Phase 3: ML Model Training & Evaluation
- ✅ Phase 4: Anti-Overfitting Verification
- ✅ Phase 5: Interactive Web Dashboard
- 🔮 Phase 6: Future Enhancements

---

## 🛠️ Technologies

### Backend / ML Pipeline
| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| NumPy | Numerical computations |
| Matplotlib | Static visualizations |
| scikit-learn | ML models (RF, LR, MLP), preprocessing, metrics |
| XGBoost | Gradient boosting classifier |
| imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| ipywidgets | Interactive Jupyter widgets |

### Frontend / Dashboard
| Technology | Purpose |
|------------|---------|
| HTML5 | Semantic structure |
| CSS3 | Glassmorphism UI, animations, responsive design |
| JavaScript (ES6+) | Physics engine, real-time simulation |
| Chart.js 4.x | 12 interactive charts and visualizations |
| Canvas API | Trajectory rendering, hero animation |
| Inter + JetBrains Mono | Typography (Google Fonts) |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🦠 BacMotionAI v2 — Anti-Overfitting Design | Interactive Dashboard**

Made with ❤️ for computational biophysics

[⬆ Back to Top](#-bacmotionai)

</div>
