"""Train and persist the Census income classifier."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    performance_on_slices,
    train_model,
)

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
LABEL = "salary"


def train_and_save(data_path, model_path, test_size=0.2):
    """Train on a CSV and save the model, encoders, and feature configuration."""
    data = pd.read_csv(data_path)
    data.columns = data.columns.str.strip()
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].str.strip()
    required_columns = set(CATEGORICAL_FEATURES + [LABEL])
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    train, test = train_test_split(
        data, test_size=test_size, random_state=42, stratify=data[LABEL]
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CATEGORICAL_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CATEGORICAL_FEATURES, label=LABEL,
        training=False, encoder=encoder, lb=lb
    )
    model = train_model(X_train, y_train)
    metrics = compute_model_metrics(y_test, inference(model, X_test))
    slices = {
        f"{feature}={value}": test[feature].eq(value).to_numpy()
        for feature in CATEGORICAL_FEATURES
        for value in sorted(test[feature].dropna().unique())
    }
    slice_metrics = performance_on_slices(model, X_test, y_test, slices)
    slice_output = Path(model_path).parent.parent / "slice_output.txt"
    with slice_output.open("w", encoding="utf-8") as output:
        for name, values in sorted(slice_metrics.items()):
            output.write(
                f"{name}: precision={values[0]:.5f}, "
                f"recall={values[1]:.5f}, fbeta={values[2]:.5f}\n"
            )
    artifact = {
        "model": model,
        "encoder": encoder,
        "label_binarizer": lb,
        "categorical_features": CATEGORICAL_FEATURES,
        "label": LABEL,
        "metrics": {"precision": metrics[0], "recall": metrics[1], "fbeta": metrics[2]},
        "slice_metrics": slice_metrics,
    }
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/census.csv")
    parser.add_argument("--model", default="model/model.joblib")
    args = parser.parse_args()
    artifact = train_and_save(args.data, args.model)
    print(f"Saved model to {args.model}")
    print(artifact["metrics"])


if __name__ == "__main__":
    main()
