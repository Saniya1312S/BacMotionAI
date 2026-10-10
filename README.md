<div align="center">

# 🦠 BacMotionAI

**AI-Powered Bacterial Motility Analysis Under Confinement & Chemical Stimuli**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-DL-006600?style=flat-square)](https://xgboost.readthedocs.io)
[![Chart.js](https://img.shields.io/badge/Chart.js-Dashboard-FF6384?style=flat-square)](https://www.chartjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Demo-Live-blue?style=flat-square)](https://Saniya1312S.github.io/BacMotionAI/)

</div>

---

## 🔬 What is BacMotionAI?

BacMotionAI is an end-to-end **Machine Learning and Deep Learning** project that simulates how *E. coli* bacteria move through confined micro-channels under different chemical conditions, then uses AI to automatically classify their behavior into **5 distinct motility regimes**. It combines physics-based simulation, feature engineering, ML/DL model training, and a live interactive web dashboard — making it a complete data science pipeline from data generation to prediction.

> **🎯 Live Demo:** [https://Saniya1312S.github.io/BacMotionAI/](https://Saniya1312S.github.io/BacMotionAI/)

---

## 📌 Problem Statement

Bacterial motility plays a critical role in infection spread, biofilm formation, drug delivery, and microfluidic device design. Manually analyzing bacterial movement patterns from microscopy data is time-consuming and subjective. This project automates motility classification using **ML/DL models** trained on physics-simulated trajectory data, achieving **~78% accuracy** with verified anti-overfitting measures.

---

## ✨ Key Features

- 🧬 **Physics-Based Simulation** — Run-and-tumble model with hydrodynamic wall interactions
- 📊 **14 Engineered Features** — Speed statistics, wall alignment, run lengths, diffusivity
- 🤖 **4 ML/DL Models** — Random Forest, Logistic Regression, XGBoost, MLP Neural Network
- 🎯 **5 Motility Regimes** — Inhibited, Wall-Guided, Chemotactic, Confined, Free-Swimming
- 🛡️ **Anti-Overfitting Design** — SMOTE on train only, no data leakage, CV-test gap < 5%
- 🎮 **Interactive Dashboard** — Real-time simulation, live charts, regime prediction
- 📈 **8 Analysis Charts** — ROC curves, confusion matrix, learning curves, feature importance

---

## 🖼️ Dashboard Screenshots

<div align="center">

| Hero & Pipeline | Interactive Simulator |
|:-:|:-:|
| ![Hero](Images/dashboard_hero.png) | ![Simulator](Images/dashboard_simulator.png) |

| Analysis Charts | ML Predictions |
|:-:|:-:|
| ![Charts](Images/dashboard_charts.png) | ![Predictions](Images/dashboard_predictions.png) |

</div>

---

## 🏗️ System Architecture

<div align="center">

![Architecture](Images/architecture_diagram.png)

*Figure: AI-Integrated Bacterial Motility Analysis Framework*

</div>

**Pipeline:** Experiment Setup → Simulate Bacterial Motion → Chemotaxis → Dataset Creation → Feature Extraction → AI/ML Modelling → Prediction

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+, JavaScript ES6+ |
| **ML/DL** | scikit-learn, XGBoost, MLP Neural Network |
| **Data** | NumPy, Pandas, SMOTE (imbalanced-learn) |
| **Visualization** | Matplotlib, Chart.js 4.x, Canvas API |
| **Explainability** | SHAP |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Chart.js |
| **Deployment** | GitHub Pages |

---

## 📊 Dataset & Features

The dataset contains **2,000 simulated bacterial trajectories** with **14 extracted features** per sample:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `mean_speed` | Average instantaneous speed (µm/s) |
| 2 | `std_speed` | Speed standard deviation |
| 3 | `q25_speed` / `q75_speed` | Speed quartiles |
| 4 | `cv_speed` | Coefficient of variation |
| 5 | `speed_skewness` / `speed_kurtosis` | Distribution shape |
| 6 | `mean_alignment` / `std_alignment` | Wall alignment |cos(θ)| |
| 7 | `wall_rate` | Wall collision frequency |
| 8 | `near_wall_frac` | Time fraction near walls |
| 9 | `mean_run_len` | Average run length between tumbles |
| 10 | `confinement_idx` | Cell radius / channel width |
| 11 | `eff_diffusivity` | Effective diffusivity (v²/2Dᵣ) |

> 📄 Full data dictionary: [docs/features_dictionary.md](docs/features_dictionary.md)

---

## 🤖 ML/DL Pipeline

```
Raw Simulation → Feature Extraction → StandardScaler (train only)
    → SMOTE Balancing (train only) → Model Training → Evaluation
```

### Model Comparison

| Model | Type | Test Accuracy | ROC-AUC | Log Loss |
|-------|------|:---:|:---:|:---:|
| 🌳 **Random Forest** | ML (Ensemble) | **~0.78** | **~0.94** | ~0.65 |
| ⚡ XGBoost | ML (Boosting) | ~0.76 | ~0.93 | ~0.70 |
| 🧠 MLP Neural Network | DL | ~0.74 | ~0.92 | ~0.78 |
| 📈 Logistic Regression | ML (Linear) | ~0.62 | ~0.87 | ~1.10 |

> **Note:** Metrics are approximate from the existing codebase. All models show CV-test gap < 5%, confirming no overfitting.

> 📄 Detailed report: [docs/project_report.md](docs/project_report.md)

---

## 🚀 Quick Start

### View the Dashboard (No Install Needed)
```bash
# Open index.html directly in your browser, or:
python -m http.server 8000
# Then visit http://localhost:8000
```

### Run the ML Pipeline
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python scripts/generate_dataset.py

# 3. Train models
python scripts/train_models.py

# 4. Run the notebook
jupyter notebook notebooks/BacMotionAI_v2_AntiOverfit_Fixed.ipynb
```

---

## 📁 Project Structure

```
BacMotionAI/
├── index.html                    # Interactive dashboard
├── style.css                     # Dashboard styling
├── app.js                        # Simulation engine + charts
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/                          # Reusable Python modules
│   ├── simulation.py             # Physics simulation engine
│   ├── features.py               # Feature extraction (14 features)
│   └── models.py                 # ML model configurations
│
├── notebooks/                    # Jupyter notebooks
│   └── BacMotionAI_v2_AntiOverfit_Fixed.ipynb
│
├── scripts/                      # Runnable scripts
│   ├── generate_dataset.py       # Generate training data
│   ├── train_models.py           # Train all 4 models
│   ├── evaluate_models.py        # Generate evaluation plots
│   └── fix_notebook.py           # Bug fix patch script
│
├── data/                         # Dataset & documentation
│   └── README.md                 # Data dictionary
│
├── models/                       # Saved model artifacts
│   └── README.md                 # Model documentation
│
├── Images/                       # Screenshots & figures
│   └── architecture_diagram.png  # System architecture
│
├── docs/                         # Detailed documentation
│   ├── project_report.md         # Full technical report
│   ├── features_dictionary.md    # Feature descriptions
│   └── dashboard_guide.md        # Dashboard user guide
│
├── tests/                        # Unit tests
│   └── test_simulation.py
│
└── .github/workflows/
    └── deploy.yml                # GitHub Pages deployment
```

---

## ⚠️ Limitations & Future Scope

**Current Limitations:**
- Dataset is simulation-based, not from real microscopy data
- 2D channel geometry only
- Browser dashboard uses a JavaScript rule-based classifier (not the trained Python model)

**Future Work:**
- 🔬 Integrate real microscopy tracking data
- 🧊 3D channel simulation
- 🧠 Deep Learning models (LSTM, CNN on trajectory images)
- 🌐 Backend API for real-time Python model inference
- 📦 PyPI package release

---

## 👩‍💻 Author

**Saniya Saratkar**  
Data Science & AI/ML Engineer  
📧 saniyasaratkar1312@gmail.com  
🔗 [GitHub](https://github.com/Saniya1312S)

---

## 📚 References

1. Berg, H.C. & Brown, D.A. (1972). *Chemotaxis in E. coli analysed by three-dimensional tracking.* Nature.
2. Berke, A.P. et al. (2008). *Hydrodynamic attraction of swimming microorganisms by surfaces.* Physical Review Letters.
3. Lauga, E. et al. (2006). *Swimming in circles: motion of bacteria near solid boundaries.* Biophysical Journal.

> **Note:** Additional academic references from the original project report require verification.

---

<div align="center">

**⭐ Star this repo if you found it useful!**

*Built with ❤️ for computational biophysics and AI/ML*

</div>
