# file: credit_card_fraud/main_simple.py
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

CSV_NAME, SEED, N_SPLITS = "creditcard.csv", 42, 5

def evaluate(y, proba, name):
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc_auc_score(y, proba):.4f} | PR-AUC: {average_precision_score(y, proba):.4f}")

    # threshold = 0.5
    pred = (proba >= 0.5).astype(int)
    print("\n-- Thr=0.5 --")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=4))

    # best F1 threshold
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    best_thr = thr[np.nanargmax(f1)] if len(thr) else 0.5
    pred = (proba >= best_thr).astype(int)
    print(f"\n-- Best F1 Thr={best_thr:.3f} --")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=4))

def main():
    # Data
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, CSV_NAME))
    X, y = df.drop(columns=["Class"]), df["Class"].values

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                class_weight="balanced_subsample",
                                random_state=SEED, n_jobs=-1)
    rf_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:,1]
    evaluate(y, rf_proba, "RandomForest")

    # XGBoost 
    if HAS_XGB:
        pos, neg = y.sum(), len(y)-y.sum()
        xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9,
                            tree_method="hist", eval_metric="aucpr",
                            scale_pos_weight=neg/pos, random_state=SEED, n_jobs=-1)
        xgb_proba = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:,1]
        evaluate(y, xgb_proba, "XGBoost")

if __name__ == "__main__":
    main()
