import io
import os
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from constants import DATA_PROCESSED_DIR, DATA_RAW_DIR, RANDOM_STATE

TRAIN_RATIO = 0.6
VALID_RATIO = 0.2
TEST_RATIO = 0.2


def download_and_extract_zip(url: str):
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    try:
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(DATA_RAW_DIR)

        print("Download ok")
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
    except zipfile.BadZipFile:
        print("Download failed: The downloaded file is not a ZIP")


def save_dataframe(df: pd.DataFrame, split_type: str):
    folder = os.path.join(DATA_PROCESSED_DIR, str(RANDOM_STATE))
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{split_type}.csv")
    df.to_csv(filepath, index=False)
    print(f"Saved {split_type} to {filepath}")


def main():
    raw_data = os.path.join(DATA_RAW_DIR, "creditcard.csv")

    if not os.path.exists(raw_data):
        download_and_extract_zip("https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud")

    df = pd.read_csv(os.path.join(DATA_RAW_DIR, "creditcard.csv"))

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE, stratify=y)

    valid_ratio_adjusted = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, train_size=valid_ratio_adjusted, random_state=RANDOM_STATE, stratify=y_temp
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_valid, y_valid], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    save_dataframe(train_df, "training")
    save_dataframe(valid_df, "validation")
    save_dataframe(test_df, "test")


if __name__ == "__main__":
    main()
