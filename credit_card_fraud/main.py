import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix
)

RANDOM_SEED = 1

def main():
    # --- Les data  ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "creditcard.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Legg 'creditcard.csv' i samme mappe som main.py.")
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # --- Train/test split (stratifisert pga. ubalanse) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    # --- Random Forest ---
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_SEED
    ).fit(X_train, y_train)

    # --- Prediksjon ---
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # --- Grunnleggende metrikker ---
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_test, y_proba)
    pr  = average_precision_score(y_test, y_proba)

    print(f"\nAccuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"ROC-AUC={roc:.4f}   PR-AUC={pr:.4f}  (PR-AUC er viktig ved ubalanse)")

    # --- Confusion matrix + prosent ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    correct = tn + tp
    wrong = fp + fn

    print("\nConfusion matrix:")
    print(cm)
    print(f"\nTotalt test-eksempler: {total}")
    print(f"Riktig: {correct} ({correct/total:.2%})")
    print(f"Feil:   {wrong} ({wrong/total:.2%})")

    # --- Feature importances til fil ---
    (pd.Series(clf.feature_importances_, index=X.columns)
       .sort_values(ascending=False)
       .to_csv(os.path.join(script_dir, "feature_importances.csv"), header=["importance"]))
    print("\nFeature importances lagret til feature_importances.csv")

if __name__ == "__main__":
    main()
