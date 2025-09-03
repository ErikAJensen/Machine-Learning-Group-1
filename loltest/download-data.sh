#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/data"

curl -L -o "$SCRIPT_DIR/data/league-of-legends-diamond-ranked-games-10-min.zip"\
  https://www.kaggle.com/api/v1/datasets/download/bobbyscience/league-of-legends-diamond-ranked-games-10-min

unzip "$SCRIPT_DIR/data/league-of-legends-diamond-ranked-games-10-min.zip" -d "$SCRIPT_DIR/data/league-of-legends-diamond-ranked-games-10-min"