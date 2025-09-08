#!/bin/bash

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"

mkdir -p "$DATA_DIR"

curl -L -o "$DATA_DIR/credit-card-fraud-detection-dataset-2023.zip"\
  https://www.kaggle.com/api/v1/datasets/download/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

unzip "$DATA_DIR/credit-card-fraud-detection-dataset-2023.zip" -d "$DATA_DIR/credit-card-fraud-detection-dataset-2023"
