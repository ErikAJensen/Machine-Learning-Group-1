# file: credit_card_fraud/main.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)

RANDOM_SEED = 1
N_ESTIMATORS = 300
MAX_DEPTH = 12
TEST_SIZE = 0.2


def main():
    # --- Les data ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "creditcard.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Legg 'creditcard.csv' i samme mappe som main.py.")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # --- Train/test split (stratifisert pga. ubalanse) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # --- Modell: Random Forest med class_weight ---
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",  # håndterer ubalanse
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    # --- Prediksjon ---
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # --- Grunnleggende metrikker ---
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("\n=== Resultater ===")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"ROC-AUC:    {roc:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f} (PR-AUC er viktig ved ubalanse)")

    # --- Confusion matrix + prosent ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    correct = tn + tp
    wrong = fp + fn

    print("\nConfusion matrix [[TN FP]\n                   [FN TP]]:\n", cm)
    print(f"\nTotalt test-eksempler: {total}")
    print(f"Riktig: {correct} ({correct/total:.2%})")
    print(f"Feil:   {wrong} ({wrong/total:.2%})")

    # --- terskel-tuning for bedre Recall ---
    precs, recs, thresh = precision_recall_curve(y_test, y_proba)
    f1_scores = (2 * precs * recs) / (precs + recs + 1e-9)
    best_idx = f1_scores.argmax()
    best_thr = thresh[max(best_idx - 1, 0)] if best_idx < len(thresh) else 0.5
    print(f"\nBeste cut-off for F1: {best_thr:.3f}  |  F1={f1_scores[best_idx]:.4f}")


if __name__ == "__main__":
    main()
