import os
from pathlib import Path
from time import time

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost er ikke installert. Kjør: pip install xgboost")

from state import RANDOM_STATE

MODEL_FILE = os.path.splitext(os.path.abspath(__file__))[0] + ".pkl"


def xgb_train(X_train, y_train, X_valid, y_valid):
    """Tren en enkel XGBoost-modell med early stopping."""

    # beregn class imbalance (scale_pos_weight)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / pos) if pos else 1.0

    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="aucpr",  # precision-recall AUC
        scale_pos_weight=scale_pos_weight,
    )

    model = XGBClassifier(**params)

    print("\nTraining XGBoost...")

    start = time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=30,
        verbose=False,
    )
    end = time()

    print(f"Training completed in {end - start:.2f} seconds")
    return model


def main():
    # finn data relativt til denne fila
    HERE = Path(__file__).resolve().parent
    DATA_DIR = HERE / "data" / "processed" / "123"

    train_df = pd.read_csv(DATA_DIR / "training.csv")
    valid_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_train, y_train = train_df.drop(columns=["Class"]), train_df["Class"]
    X_valid, y_valid = valid_df.drop(columns=["Class"]), valid_df["Class"]
    X_test, y_test = test_df.drop(columns=["Class"]), test_df["Class"]

    model = xgb_train(X_train, y_train, X_valid, y_valid)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Set Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"AUPRC:     {average_precision_score(y_test, y_proba):.4f}")
    print(f"AUROC:     {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
