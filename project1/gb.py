import os
from itertools import product
from time import time

import joblib
import pandas as pd
import sklearn
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

from constants import DATA_PROCESSED_DIR, MODELS_DIR, RANDOM_STATE

ES_MODEL_FILE = os.path.join(MODELS_DIR, str(RANDOM_STATE), "gb.es.pkl")
ADASYN_MODEL_FILE = os.path.join(MODELS_DIR, str(RANDOM_STATE), "gb.adasyn.pkl")

sklearn.set_config(enable_metadata_routing=True)


def xgboost_model(X_train, y_train, X_val, y_val):
    print("\nXGBClassifier with hyperparameter tuning and early stopping")

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="aucpr",
        early_stopping_rounds=20,
    )

    xgb.set_fit_request(eval_set=True, verbose=True)

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

    pipeline = Pipeline(
        [
            ("clf", xgb),
        ]
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        scoring=f2_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        n_iter=600,
        random_state=RANDOM_STATE,
    )

    

    start = time()
    search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    end = time()

    print(f"XGB1 Randomized search completed in {end - start:.2f} seconds")
    print(f"Best hyperparameters: {search.best_params_}")



    xgb2 = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="aucpr",
    )

    pipeline2 = Pipeline(
        [
            (
                "scaler",
                ColumnTransformer(
                    transformers=[("amount_scaler", StandardScaler(), ["Amount"])],
                    remainder="passthrough",
                ),
            ),
            ("adasyn", ADASYN(random_state=RANDOM_STATE)),
            ("clf", xgb2),
        ],
    )

    search2 = RandomizedSearchCV(
        pipeline2,
        param_distributions=param_grid,
        scoring=f2_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        n_iter=200,
        random_state=RANDOM_STATE,
    )

    start = time()
    search2.fit(X_train, y_train)
    end = time()

    print(f"XGB2 Randomized search completed in {end - start:.2f} seconds")
    print(f"Best hyperparameters: {search2.best_params_}")


    return search.best_estimator_, search2.best_estimator_


def main():
    train_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "training.csv"))
    val_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "validation.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE), "test.csv"))

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_val, y_val = val_df.drop(columns=["Class"]), val_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    best_model_es, best_model_adasyn = xgboost_model(X_train, y_train, X_val, y_val)

    y_pred = best_model_es.predict(X_test)
    y_proba = best_model_es.predict_proba(X_test)[:, 1]

    print("\nTest Set Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUPRC:     {average_precision_score(y_test, y_proba):.4f}")
    print(f"AUROC:     {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(best_model_es, ES_MODEL_FILE)

    joblib.dump(best_model_adasyn, ADASYN_MODEL_FILE)



if __name__ == "__main__":
    main()
