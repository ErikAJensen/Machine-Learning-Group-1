# random_forest.py
import os
from time import time
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, fbeta_score,
    make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from state import RANDOM_STATE

MODEL_FILE = os.path.splitext(os.path.abspath(__file__))[0] + ".pkl"

def random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
    param_grid = {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
        # evt.: "max_features": ["sqrt", "log2", None],
    }
    f2_scorer = make_scorer(fbeta_score, beta=2)
    grid = GridSearchCV(rf, param_grid, scoring=f2_scorer, cv=5, n_jobs=-1, verbose=1)

    print("\nRandomForestClassifier")
    start = time(); grid.fit(X_train, y_train); end = time()
    print(f"Grid search completed in {end - start:.2f} seconds")
    print("Best hyperparameters:", grid.best_params_)
    return grid.best_estimator_

def main():
    train_df = pd.read_csv("data/processed/123/training.csv")
    valid_df = pd.read_csv("data/processed/123/validation.csv")
    test_df  = pd.read_csv("data/processed/123/test.csv")

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_valid, y_valid = valid_df.drop(columns=["Class"]), valid_df["Class"]
    X_test,  y_test  = test_df.drop(columns=["Class"]),  test_df["Class"]

    X_grid = pd.concat([X_train, X_valid])
    y_grid = pd.concat([y_train, y_valid])

    best_model = random_forest(X_grid, y_grid)

    y_pred  = best_model.predict(X_test)
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
