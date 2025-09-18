# file: credit_card_fraud/main.py
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, recall_score,
    confusion_matrix, accuracy_score
)

RANDOM_SEED   = 1
N_SPLITS      = 5
N_ESTIMATORS  = 300
MAX_DEPTH     = 12

def main():
    # --- Les data (samme mappe som main.py) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "creditcard.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Legg 'creditcard.csv' i samme mappe som main.py.")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # --- Modell: Random Forest med class_weight for ubalanse ---
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    # --- Stratified K-Fold ---
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # --- CV-scorer (mean ± std) ---
    pr_auc  = cross_val_score(clf, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    roc_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc",           n_jobs=-1)
    f1      = cross_val_score(clf, X, y, cv=cv, scoring="f1",                n_jobs=-1)
    recall  = cross_val_score(clf, X, y, cv=cv, scoring="recall",            n_jobs=-1)

    print("\n=== Stratified K-Fold ({} folds) ===".format(N_SPLITS))
    print(f"PR-AUC:   {pr_auc.mean():.4f} ± {pr_auc.std():.4f}")
    print(f"ROC-AUC:  {roc_auc.mean():.4f} ± {roc_auc.std():.4f}")
    print(f"F1:       {f1.mean():.4f} ± {f1.std():.4f}")
    print(f"Recall:   {recall.mean():.4f} ± {recall.std():.4f}")

    # --- Samlet OOF-prediksjon for confusion matrix og prosent riktig/feil ---
    # Predict (labels) via CV
    y_pred_oof = cross_val_predict(clf, X, y, cv=cv, method="predict", n_jobs=-1)
    # Proba kan være nyttig om du senere vil terskel-tune
    # y_proba_oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    cm = confusion_matrix(y, y_pred_oof)
    tn, fp, fn, tp = cm.ravel()
    total   = cm.sum()
    correct = tn + tp
    wrong   = fp + fn

    print("\nConfusion matrix (OOF over alle folds) [[TN FP]\n                                   [FN TP]]:")
    print(cm)
    print(f"\nTotalt:  {total}")
    print(f"Riktig:  {correct} ({correct/total:.2%})")
    print(f"Feil:    {wrong} ({wrong/total:.2%})")

    # Ekstra samlet mål (OOF):
    acc_oof  = accuracy_score(y, y_pred_oof)
    f1_oof   = f1_score(y, y_pred_oof, average="binary", zero_division=0)
    rec_oof  = recall_score(y, y_pred_oof, average="binary", zero_division=0)
    print(f"\nOOF Accuracy: {acc_oof:.4f} | OOF F1: {f1_oof:.4f} | OOF Recall: {rec_oof:.4f}")

if __name__ == "__main__":
    main()
