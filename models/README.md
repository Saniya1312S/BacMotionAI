# 🤖 Trained Model Artifacts

## Models
| File | Model | Type |
|------|-------|------|
| `random_forest.joblib` | Random Forest | ML (Ensemble) |
| `logistic_regression.joblib` | Logistic Regression | ML (Linear) |
| `xgboost.joblib` | XGBoost | ML (Boosting) |
| `mlp.joblib` | MLP Neural Network | DL |
| `scaler.joblib` | StandardScaler | Preprocessing |
| `label_encoder.joblib` | LabelEncoder | Preprocessing |

## How to Retrain
```bash
# 1. Generate dataset first
python scripts/generate_dataset.py

# 2. Train all models
python scripts/train_models.py

# 3. Evaluate and generate plots
python scripts/evaluate_models.py
```

## Anti-Overfitting Measures
- StandardScaler fit on **training set only**
- SMOTE applied to **training set only**
- Stratified train/test split (80/20)
- 5-fold cross-validation
- CV-test accuracy gap verified < 5%

> **Note:** Model `.joblib` files are not tracked in git. Regenerate them using the commands above.
