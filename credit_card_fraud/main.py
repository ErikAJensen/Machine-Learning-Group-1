# file: credit_card_fraud/main.py
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, roc_auc_score
)

# ==== Enkle innstillinger (Kan endre ved behov) ====
CSV_NAME   = "creditcard.csv"
N_SPLITS   = 5
N_TREES    = 150
MAX_DEPTH  = 8
RNG_SEED   = 1

# Gjør modellen mer sensitiv:
POS_WEIGHT_MULT = 2.0   # 1.0 = bare "balanced". Øk (2.0, 3.0) for mer vekt på fraud
FIXED_THRESH    = 0.50  # vi rapporterer også en fast terskel
# =====================================================

def class_weight_balanced_plus(y, pos_mult=1.0):
    n = len(y); n_pos = y.sum(); n_neg = n - n_pos
    w0 = n / (2.0 * n_neg)
    w1 = (n / (2.0 * n_pos)) * pos_mult
    return {0: w0, 1: w1}

def print_block(title, y_true, y_pred, y_proba=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    print(f"\n=== {title} ===")
    print(f"Confusion [[TN FP]; [FN TP]]:\n[[{tn} {fp}]\n [{fn} {tp}]]")
    print(f"Correct: {tn+tp:,} ({(tn+tp)/total:.2%}) | Wrong: {fp+fn:,} ({(fp+fn)/total:.2%})")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f} | "
          f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f} | "
          f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f} | "
          f"F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if y_proba is not None:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f} | "
              f"PR-AUC: {average_precision_score(y_true, y_proba):.4f}")

def main():
    # --- data ---
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, CSV_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing '{CSV_NAME}' next to this script.")
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int).values

    # --- modell (RF med ekstra positiv vekt) ---
    cw = class_weight_balanced_plus(y, pos_mult=POS_WEIGHT_MULT)
    clf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH,
        class_weight=cw, n_jobs=-1, random_state=RNG_SEED
    )

    # --- CV-oppsett + kjapp rapport ---
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG_SEED)
    scores = cross_validate(
        clf, X, y, cv=cv, n_jobs=-1,
        scoring={"recall":"recall", "f1":"f1", "roc_auc":"roc_auc", "pr_auc":"average_precision"}
    )
    print("\n=== CV (mean ± std) ===")
    for k, lab in [("recall","Recall"), ("f1","F1"), ("roc_auc","ROC-AUC"), ("pr_auc","PR-AUC")]:
        v = scores[f"test_{k}"]
        print(f"{lab:7s}: {v.mean():.4f} ± {v.std():.4f}")

    # --- OOF proba for terskler ---
    y_proba_oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    # 1) Fast terskel (f.eks. 0.50)
    y_pred_fixed = (y_proba_oof >= FIXED_THRESH).astype(int)
    print_block(f"OOF @ threshold = {FIXED_THRESH:.2f}", y, y_pred_fixed, y_proba=y_proba_oof)

    # 2) Automatisk beste F1-terskel
    prec, rec, thr = precision_recall_curve(y, y_proba_oof)
    f1s = 2*(prec*rec)/(prec+rec+1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    y_pred_best = (y_proba_oof >= best_thr).astype(int)
    print_block(f"OOF @ best F1 threshold = {best_thr:.4f}", y, y_pred_best, y_proba=y_proba_oof)

if __name__ == "__main__":
    main()
