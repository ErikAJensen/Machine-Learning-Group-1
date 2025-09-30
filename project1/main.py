import os

import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from constants import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_STATE


def print_metrics(name, y_test, y_pred, y_proba):
    print(f"\n{name} Test Set Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUPRC:     {average_precision_score(y_test, y_proba):.4f}")
    print(f"AUROC:     {roc_auc_score(y_test, y_proba):.4f}")


def main():
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "training.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "test.csv"))

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    cart_base = DecisionTreeClassifier(random_state=RANDOM_STATE)

    cart_base.fit(X_train, y_train)

    cart_adasyn = Pipeline(
        [
            (
                "scaler",
                ColumnTransformer(transformers=[("amount_scaler", StandardScaler(), ["Amount"])], remainder="passthrough"),
            ),
            ("adasyn", ADASYN(random_state=RANDOM_STATE)),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )

    cart_adasyn.fit(X_train, y_train)

    cart_ht = joblib.load(os.path.join(MODELS_DIR, str(RANDOM_STATE), "cart.pkl"))

    print_metrics("CART Base", y_test, cart_base.predict(X_test), cart_base.predict_proba(X_test)[:, 1])
    print_metrics("CART with ADASYN", y_test, cart_adasyn.predict(X_test), cart_adasyn.predict_proba(X_test)[:, 1])
    print_metrics(
        "CART with ADASYN Hyperparameter Tuned", y_test, cart_ht.predict(X_test), cart_ht.predict_proba(X_test)[:, 1]
    )

    rf_base = RandomForestClassifier(random_state=RANDOM_STATE)

    rf_base.fit(X_train, y_train)

    rf_adasyn = Pipeline(
        [
            (
                "scaler",
                ColumnTransformer(
                    transformers=[("amount_scaler", StandardScaler(), ["Amount"])],
                    remainder="passthrough",
                ),
            ),
            ("adasyn", ADASYN(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )

    rf_adasyn.fit(X_train, y_train)

    rf_ht = joblib.load(os.path.join(MODELS_DIR, str(RANDOM_STATE), "random_forest.pkl"))

    print_metrics("Random Forest Base", y_test, rf_base.predict(X_test), rf_base.predict_proba(X_test)[:, 1])
    print_metrics("Random Forest with ADASYN", y_test, rf_adasyn.predict(X_test), rf_adasyn.predict_proba(X_test)[:, 1])
    print_metrics(
        "Random Forest with ADASYN Hyperparameter Tuned", y_test, rf_ht.predict(X_test), rf_ht.predict_proba(X_test)[:, 1]
    )


if __name__ == "__main__":
    main()
