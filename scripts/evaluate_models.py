"""
Evaluate trained models and generate visualization plots.
Run: python scripts/evaluate_models.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from src.features import FEATURE_COLS

def main():
    base = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base, 'data', 'bacterial_motility_dataset.csv')
    models_dir = os.path.join(base, 'models')
    img_dir = os.path.join(base, 'Images')
    os.makedirs(img_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("❌ Dataset not found. Run generate_dataset.py first"); return

    df = pd.read_csv(data_path)
    scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
    le = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))

    X = df[FEATURE_COLS].values
    y = le.transform(df['regime'])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = scaler.transform(X_test)

    model_files = {
        'Random Forest': 'random_forest.joblib',
        'XGBoost': 'xgboost.joblib',
        'MLP': 'mlp.joblib',
        'Logistic Regression': 'logistic_regression.joblib',
    }

    # Confusion matrix for best model (RF)
    rf_path = os.path.join(models_dir, 'random_forest.joblib')
    if os.path.exists(rf_path):
        rf = joblib.load(rf_path)
        y_pred = rf.predict(X_test_sc)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                               display_labels=le.classes_).plot(ax=ax, cmap='Blues')
        ax.set_title('Confusion Matrix — Random Forest', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        print("✅ Saved confusion_matrix.png")

        # Feature importance
        imps = rf.feature_importances_
        idx = np.argsort(imps)[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([FEATURE_COLS[i] for i in idx], imps[idx], color='#06B6D4')
        ax.set_xlabel('Gini Importance')
        ax.set_title('Feature Importances — Random Forest', fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'feature_importance.png'), dpi=150)
        plt.close()
        print("✅ Saved feature_importance.png")

    print("\n✅ Evaluation complete. Plots saved to Images/")

if __name__ == '__main__':
    main()
