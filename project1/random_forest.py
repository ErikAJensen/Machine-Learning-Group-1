import os
from time import time

import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from constants import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_STATE

MODEL_FILE = os.path.join(MODELS_DIR, str(RANDOM_STATE), "random_forest.pkl")


def random_forest(X, y):
    print("\nRandomForestClassifier with hyperparameter tuning")

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    pipeline = Pipeline(
        [
            (
                "scaler",
                ColumnTransformer(
                    transformers=[("amount_scaler", StandardScaler(), ["Amount"])],
                    remainder="passthrough",
                ),
            ),
            ("adasyn", ADASYN(random_state=RANDOM_STATE)),
            ("clf", rf),
        ]
    )

    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [5, 10, 15, 20, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_features": ["sqrt", "log2", None],
    }

    f2_scorer = make_scorer(fbeta_score, beta=2)

    grid_search = RandomizedSearchCV(
        pipeline, param_grid, scoring=f2_scorer, cv=5, n_jobs=-1, verbose=1, n_iter=50, random_state=RANDOM_STATE
    )

    start = time()
    grid_search.fit(X, y)
    end = time()

    print(f"Grid search completed in {end - start:.2f} seconds")
    print(f"Best hyperparameters: {grid_search.best_params_}")

    return grid_search.best_estimator_


def main():
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "training.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "test.csv"))

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    best_model = random_forest(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nTest Set Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUPRC:     {average_precision_score(y_test, y_proba):.4f}")
    print(f"AUROC:     {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(best_model, MODEL_FILE)


if __name__ == "__main__":
    main()
