"""
Train all 4 ML models on the bacterial motility dataset.
Run: python scripts/train_models.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, classification_report
from imblearn.over_sampling import SMOTE

from src.features import FEATURE_COLS
from src.models import build_random_forest, build_logistic_regression, build_xgboost, build_mlp, REGIME_LABELS

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'bacterial_motility_dataset.csv')
    if not os.path.exists(data_path):
        print("❌ Dataset not found. Run: python scripts/generate_dataset.py first")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")

    le = LabelEncoder()
    X = df[FEATURE_COLS].values
    y = le.fit_transform(df['regime'])

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale (fit on train only — no data leakage)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # SMOTE on train only
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
    print(f"Train (after SMOTE): {X_train_sm.shape}, Test: {X_test_sc.shape}")

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save scaler and encoder
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(models_dir, 'label_encoder.joblib'))

    builders = {
        'Random Forest': build_random_forest,
        'Logistic Regression': build_logistic_regression,
        'XGBoost': build_xgboost,
        'MLP': build_mlp,
    }

    results = []
    for name, build_fn in builders.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)

        clf = build_fn()
        if clf is None:
            print(f"  Skipped (dependency missing)")
            continue

        if name == 'XGBoost':
            X_xgb_tr, X_xgb_val, y_xgb_tr, y_xgb_val = train_test_split(
                X_train_sm, y_train_sm, test_size=0.15, random_state=42
            )
            clf.fit(X_xgb_tr, y_xgb_tr,
                    eval_set=[(X_xgb_val, y_xgb_val)],
                    verbose=False)
        else:
            clf.fit(X_train_sm, y_train_sm)

        y_pred = clf.predict(X_test_sc)
        y_proba = clf.predict_proba(X_test_sc)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        ll = log_loss(y_test, y_proba)
        cv = cross_val_score(clf, X_train_sm, y_train_sm, cv=5, scoring='accuracy')

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  Log Loss:  {ll:.4f}")
        print(f"  CV (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"  Overfit gap: {cv.mean() - acc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

        fname = name.lower().replace(' ', '_') + '.joblib'
        joblib.dump(clf, os.path.join(models_dir, fname))
        results.append({'Model': name, 'Accuracy': acc, 'ROC-AUC': auc, 'LogLoss': ll})

    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['Model']:25s}  Acc={r['Accuracy']:.4f}  AUC={r['ROC-AUC']:.4f}  LL={r['LogLoss']:.4f}")
    print(f"\n✅ Models saved to {models_dir}/")

if __name__ == '__main__':
    main()
