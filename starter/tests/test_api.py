import pandas as pd
from fastapi.testclient import TestClient

import main
from starter.ml.data import process_data
from starter.ml.model import train_model


def _artifact():
    row = pd.DataFrame(
        {
            "age": [39, 52], "workclass": ["State-gov", "Private"],
            "fnlgt": [77516, 209642], "education": ["Bachelors", "HS-grad"],
            "education-num": [13, 9], "marital-status": ["Never-married", "Married-civ-spouse"],
            "occupation": ["Adm-clerical", "Exec-managerial"],
            "relationship": ["Not-in-family", "Husband"], "race": ["White", "White"],
            "sex": ["Male", "Male"], "capital-gain": [2174, 0], "capital-loss": [0, 0],
            "hours-per-week": [40, 50], "native-country": ["United-States", "United-States"],
            "salary": ["<=50K", ">50K"],
        }
    )
    categories = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    features, labels, encoder, lb = process_data(row, categories, "salary")
    return {
        "model": train_model(features, labels),
        "encoder": encoder,
        "label_binarizer": lb,
        "categorical_features": categories,
    }


def setup_module():
    main._artifact = _artifact()


client = TestClient(main.app)


def payload():
    return {
        "age": 39, "workclass": "State-gov", "fnlgt": 77516,
        "education": "Bachelors", "education-num": 13,
        "marital-status": "Never-married", "occupation": "Adm-clerical",
        "relationship": "Not-in-family", "race": "White", "sex": "Male",
        "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
        "native-country": "United-States",
    }


def test_get_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_post_predict_returns_a_prediction():
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_returns_the_other_prediction():
    data = payload()
    data.update({
        "age": 52,
        "workclass": "Private",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "capital-gain": 0,
        "hours-per-week": 50,
    })
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_post_predict_accepts_unknown_categories():
    data = payload()
    data["workclass"] = "Unknown-category"
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
