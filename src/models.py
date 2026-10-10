"""
BacMotionAI — ML Model Configurations

Defines the 4 machine learning models used for motility regime classification:
Random Forest, Logistic Regression, XGBoost, and MLP Neural Network.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# 5 motility regime labels
REGIME_LABELS = [
    'Inhibited',
    'Wall-Guided',
    'Chemotactic',
    'Confined',
    'Free-Swimming',
]


def build_random_forest() -> RandomForestClassifier:
    """
    Random Forest classifier with regularization to prevent overfitting.
    Best performing model (~78% accuracy, ~0.94 ROC-AUC).
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )


def build_logistic_regression() -> LogisticRegression:
    """
    Regularized Logistic Regression (baseline linear model).
    ~62% accuracy, ~0.87 ROC-AUC.
    """
    return LogisticRegression(
        C=0.5,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
    )


def build_xgboost():
    """
    XGBoost gradient boosting classifier with early stopping support.
    ~76% accuracy, ~0.93 ROC-AUC.
    
    Returns None if xgboost is not installed.
    """
    if not HAS_XGB:
        print("Warning: xgboost not installed. Skipping XGBoost model.")
        return None

    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )


def build_mlp() -> MLPClassifier:
    """
    Multi-Layer Perceptron (MLP) neural network classifier.
    ~74% accuracy, ~0.92 ROC-AUC.
    Uses early stopping and L2 regularization.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.01,           # L2 regularization
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
