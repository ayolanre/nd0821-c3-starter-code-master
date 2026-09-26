import numpy as np
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    performance_on_slices,
    train_model,
)


def sample_data():
    return pd.DataFrame(
        {
            "age": [25, 45, 30, 55],
            "workclass": ["Private", "State-gov", "Private", "Federal-gov"],
            "education-num": [9, 13, 10, 14],
            "sex": ["Male", "Female", "Male", "Female"],
            "salary": ["<=50K", ">50K", "<=50K", ">50K"],
        }
    )


def test_process_data_encodes_features_and_labels():
    features, labels, encoder, lb = process_data(
        sample_data(), categorical_features=["workclass", "sex"], label="salary"
    )
    assert features.shape[0] == 4
    assert labels.shape == (4,)
    assert encoder.categories_
    assert set(lb.classes_) == {"<=50K", ">50K"}


def test_model_training_and_inference_return_predictions():
    model = train_model(np.array([[0.0], [1.0], [0.1], [0.9]]), np.array([0, 1, 0, 1]))
    predictions = inference(model, np.array([[0.05], [0.95]]))
    assert predictions.shape == (2,)
    assert set(predictions).issubset({0, 1})


def test_compute_model_metrics_returns_precision_recall_fbeta():
    metrics = compute_model_metrics(np.array([0, 1, 1]), np.array([0, 1, 0]))
    assert len(metrics) == 3
    assert all(0.0 <= metric <= 1.0 for metric in metrics)


def test_performance_on_slices_returns_metrics_for_nonempty_slice():
    model = train_model(np.array([[0.0], [1.0], [0.1], [0.9]]), np.array([0, 1, 0, 1]))
    results = performance_on_slices(
        model,
        np.array([[0.0], [1.0], [0.1], [0.9]]),
        np.array([0, 1, 0, 1]),
        {"low": np.array([True, False, True, False])},
    )
    assert "low" in results
    assert len(results["low"]) == 3
