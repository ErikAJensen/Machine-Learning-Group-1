import os
from itertools import product
from time import time

import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
from xgboost import XGBClassifier

from constants import DATA_PROCESSED_DIR, RANDOM_STATE

MODEL_FILE_NO_EARLY_STOPPING = os.path.splitext(os.path.abspath(__file__))[0] + "-noes.pkl"
MODEL_FILE_EARLY_STOPPING = os.path.splitext(os.path.abspath(__file__))[0] + "-es.pkl"


def xgboost_model(X_train, y_train, X_val, y_val):
    print("\nXGBClassifier with hyperparameter tuning and early stopping")

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="aucpr",
    )

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
            ("clf", xgb),
        ]
    )

    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 5, 7, 10],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__gamma": [0, 1, 5],
        "clf__reg_alpha": [0, 0.1, 1],
        "clf__reg_lambda": [1, 1.5, 2],
    }

    f2_scorer = make_scorer(fbeta_score, beta=2)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        scoring=f2_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        n_iter=200,
        random_state=RANDOM_STATE,
    )

    start = time()
    search.fit(X_train, y_train)
    end = time()

    print(f"Randomized search completed in {end - start:.2f} seconds")
    print(f"Best hyperparameters: {search.best_params_}")

    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

    preprocessor = ColumnTransformer(
        transformers=[("amount_scaler", StandardScaler(), ["Amount"])],
        remainder="passthrough",
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    ada = ADASYN(random_state=RANDOM_STATE)
    X_train_res, y_train_res = ada.fit_resample(X_train_proc, y_train)

    best_xgb = XGBClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric="aucpr", **best_params, early_stopping_rounds=20
    )

    best_xgb.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_val_proc, y_val)],
        verbose=False,
    )

    final_pipeline = Pipeline(
        [
            ("scaler", preprocessor),
            ("adasyn", ada),
            ("clf", best_xgb),
        ]
    )

    return search.best_estimator_, final_pipeline


def main():
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "training.csv"))
    val_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "validation.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "test.csv"))

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_val, y_val = val_df.drop(columns=["Class"]), val_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    a, b = xgboost_model(X_train, y_train, X_val, y_val)

    a_y_pred = a.predict(X_test)
    b_y_pred = b.predict(X_test)
    a_y_proba = a.predict_proba(X_test)[:, 1]
    b_y_proba = b.predict_proba(X_test)[:, 1]

    print("\nTest Set Metrics (No Early Stopping):")
    print(f"Accuracy:  {accuracy_score(y_test, a_y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, a_y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, a_y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, a_y_pred):.4f}")
    print(f"AUPRC:     {average_precision_score(y_test, a_y_proba):.4f}")
    print(f"AUROC:     {roc_auc_score(y_test, a_y_proba):.4f}")

    print("Test Set Metrics (With Early Stopping):")
    print(f"Accuracy:  {accuracy_score(y_test, b_y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, b_y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, b_y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, b_y_pred):.4f}")
    print(f"AUPRC:     {average_precision_score(y_test, b_y_proba):.4f}")
    print(f"AUROC:     {roc_auc_score(y_test, b_y_proba):.4f}")

    joblib.dump(a, MODEL_FILE_NO_EARLY_STOPPING)
    joblib.dump(b, MODEL_FILE_EARLY_STOPPING)


if __name__ == "__main__":
    main()
