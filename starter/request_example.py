"""Send one prediction request to a running local or deployed API."""

import os

import requests


payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

response = requests.post(
    f"{os.getenv('API_URL', 'http://localhost:8000')}/predict",
    json=payload,
    timeout=30,
)
response.raise_for_status()
print(response.json())
print(response.status_code)
