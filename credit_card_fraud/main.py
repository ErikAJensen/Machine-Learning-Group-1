# file: credit_card_fraud/main_simple.py
import sys, platform
print(f"[Python] {sys.executable}")
print(f"[Version] {platform.python_version()}")
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

CSV_NAME, SEED, N_SPLITS = "creditcard.csv", 42, 5


# -------------------------------------------------
# Hjelpefunksjon: finn threshold som gir best F1
# -------------------------------------------------
def best_f1_threshold(y, proba):
    p, r, thr = precision_recall_curve(y, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    return thr[i] if i < len(thr) else 0.50


# -------------------------------------------------
# Evalueringsfunksjon: skriver ut ROC/PR + confusion matrix
# -------------------------------------------------
def print_eval(y, proba, name):
    roc, pr = roc_auc_score(y, proba), average_precision_score(y, proba)
    print("\n" + "=" * 55)
    print(f"{name}  |  OOF ROC-AUC: {roc:.4f}  |  OOF PR-AUC: {pr:.4f}")

    # To terskler: standard 0.5 og best F1
    for title, thr in [
        ("Threshold = 0.50", 0.50),
        ("Best F1 threshold", best_f1_threshold(y, proba))
    ]:
        yhat = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
        total = tn + fp + fn + tp
        correct, wrong = tn + tp, fp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        print(f"\n-- {title} (thr={thr:.3f}) --")
        print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
        print(f"Correct={correct:,} ({correct/total:.2%})  |  Wrong={wrong:,} ({wrong/total:.2%})")
        print(f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print("=" * 55)


# -------------------------------------------------
# Hovedfunksjon
# -------------------------------------------------
def main():
    # --- Data ---
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(here, CSV_NAME))
    X, y = df.drop(columns=["Class"]), df["Class"].astype(int).values

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300,        # <-- juster antall trær
        max_depth=10,             # <-- juster dybden 
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )
    rf_proba = cross_val_predict(
        rf, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    print_eval(y, rf_proba, "RandomForest")

    # --- XGBoost (valgfritt hvis installert) ---
    if HAS_XGB:
        pos, neg = y.sum(), len(y) - y.sum()
        # parametre
        xgb = XGBClassifier(
            n_estimators=400,          # <-- juster antall boosting-runder
            max_depth=4,               # <-- juster tre-dybde
            learning_rate=0.05,        # <-- lavere = tregere men ofte bedre
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=neg/pos,  # balanserer klassene
            random_state=SEED,
            n_jobs=-1
        )
        xgb_proba = cross_val_predict(
            xgb, X, y, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        print_eval(y, xgb_proba, "XGBoost")
    else:
        print("\n[Info] xgboost ikke installert – hopper over XGBoost.")


if __name__ == "__main__":
    main()
