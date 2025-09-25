import os
from time import time
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from state import RANDOM_STATE


MODEL_FILE = os.path.splitext(os.path.abspath(__file__))[0] + ".pkl"


def cart(X_train, y_train):
    dtree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    param_grid = {
        "max_depth": [5, 10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 3, 4, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    f2_scorer = make_scorer(fbeta_score, beta=2)

    grid_search = GridSearchCV(dtree, param_grid, scoring=f2_scorer, cv=5, n_jobs=-1, verbose=1)

    print("\nDecisionTreeClassifier")

    start = time()
    grid_search.fit(X_train, y_train)
    end = time()
    best_model = grid_search.best_estimator_

    print(f"Grid search completed in {end - start:.2f} seconds")
    print("Best hyperparameters:", grid_search.best_params_)

    return best_model


def main():
    train_df = pd.read_csv("data/processed/123/training.csv")
    valid_df = pd.read_csv("data/processed/123/validation.csv")
    test_df = pd.read_csv("data/processed/123/test.csv")

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_valid, y_valid = valid_df.drop(columns=["Class"]), valid_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    X_grid = pd.concat([X_train, X_valid])
    y_grid = pd.concat([y_train, y_valid])

    best_model = cart(X_grid, y_grid)

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
