import importlib
import io
import os
import shutil
import sys
import zipfile

import requests
from sklearn.model_selection import train_test_split

from constants import DATA_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR, RANDOM_STATE


def download_and_extract_zip(url: str):
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    try:
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(DATA_RAW_DIR)

        print("Download ok")
    except Exception as e:
        print(f"Download failed: {e}")


def split_data(all_image_paths, labels, random_state):
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_image_paths, labels, test_size=0.3, random_state=random_state, stratify=labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    return {"training": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}


def write_processed_data(split_data_dict, processed_base_dir, random_seed):
    output_root_dir = os.path.join(processed_base_dir, str(random_seed))

    if os.path.exists(output_root_dir):
        shutil.rmtree(output_root_dir)

    os.makedirs(output_root_dir)

    for split_name, (image_paths, labels) in split_data_dict.items():
        print(f"Processing {split_name} set...")
        split_dir = os.path.join(output_root_dir, split_name)

        for i, img_path in enumerate(image_paths):
            label = labels[i]
            target_class_dir = os.path.join(split_dir, label)
            os.makedirs(target_class_dir, exist_ok=True)

            img_filename = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(target_class_dir, img_filename))

            if (i + 1) % 1000 == 0:
                print(f"  Copied {i + 1}/{len(image_paths)} images for {split_name}")


def main():
    if not os.path.exists(DATA_RAW_DIR):
        download_and_extract_zip("https://www.kaggle.com/api/v1/datasets/download/alessiocorrado99/animals10")
    else:
        print("Data already exists. Skipping download.")

    sys.path.append("./data/raw")
    translate = getattr(importlib.import_module("translate"), "translate")

    all_image_paths = []
    labels = []

    raw_images_dir = os.path.join(DATA_RAW_DIR, "raw-img")
    temp_renamed_dir = os.path.join(DATA_DIR, "tmp")
    if os.path.exists(temp_renamed_dir):
        shutil.rmtree(temp_renamed_dir)
    shutil.copytree(raw_images_dir, temp_renamed_dir)

    for class_folder_italian in os.listdir(temp_renamed_dir):
        old_path = os.path.join(temp_renamed_dir, class_folder_italian)

        if os.path.isdir(old_path):
            class_folder_english = translate.get(class_folder_italian)
            if class_folder_english is None:
                continue
            new_path = os.path.join(temp_renamed_dir, class_folder_english)
            os.rename(old_path, new_path)

            for img_name in os.listdir(new_path):
                all_image_paths.append(os.path.join(new_path, img_name))
                labels.append(class_folder_english)

    split_results = split_data(all_image_paths, labels, RANDOM_STATE)
    for split_name, (paths, _) in split_results.items():
        print(f"  {split_name.capitalize()} set: {len(paths)} images")

    write_processed_data(split_results, DATA_PROCESSED_DIR, RANDOM_STATE)

    if os.path.exists(temp_renamed_dir):
        shutil.rmtree(temp_renamed_dir)


if __name__ == "__main__":
    main()
