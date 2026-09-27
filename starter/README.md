Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* **Option 1: Using pip and venv (Recommended)**
    * Ensure you have Python 3.13 installed
    * Create virtual environment: `python3.13 -m venv .venv`
    * Activate environment: `source .venv/bin/activate` (On Windows: `.venv\Scripts\activate`)
    * Install dependencies: `pip install -r requirements.txt`

* **Option 2: Using conda**
    * Download and install conda if you don't have it already.
    * conda create -n [envname] "python=3.13" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn pydantic httpx matplotlib seaborn -c conda-forge
    * Install git either through conda ("conda install git") or through your CLI, e.g. sudo apt-get git.

## Repositories

* Create a directory for the project and initialize Git and DVC.
   * As you work on the code, continually commit changes. Trained models you want to keep must be committed to DVC.
* Connect your local Git repository to GitHub.

## DVC Remote

This project uses the Google Drive DVC remote configured in `.dvc/config`. Authenticate with the Google account that can access the shared folder, then use `dvc pull -r gdrive` to retrieve artifacts or `dvc push -r gdrive` to upload them. Keep OAuth credentials and token caches out of Git.

## GitHub Actions

* Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
   * Make sure you set up the GitHub Action to use Python 3.13 (same version as development).
   * Note: Add flake8 to requirements.txt if you want to use it for linting: `pip install flake8`
* The checked-in workflow runs pytest and flake8 on Python 3.13. It does not need Google Drive credentials because its tests use local test fixtures.

## Data

* Download census.csv from the data folder in the starter repository.
   * Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.
* Track the Census data and model with DVC, then push them to the configured Google Drive remote.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.
* Commit this modified data to DVC under a new name (we often want to keep the raw data untouched but then can keep updating the cooked version).

## Model

* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
   * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

The API is publicly deployed on Render at:

https://nd0821-c3-starter-code-master-vbpv.onrender.com

Render uses `starter/` as its root directory, installs the app and DVC dependencies,
and pulls the model from Google Drive before starting Uvicorn. To send the example
prediction request from this directory in PowerShell:

```powershell
$env:API_URL = "https://nd0821-c3-starter-code-master-vbpv.onrender.com"
python request_example.py
```

## Completed project workflow

The implementation in this repository provides a reproducible Census income
classifier and FastAPI service. From this `starter` directory:

```bash
python -m pip install -r requirements.txt
python -m starter.train_model --data data/census.csv --model model/model.joblib
uvicorn main:app --host 0.0.0.0 --port 8000
python request_example.py
python -m pytest -q
```

The training command keeps the raw CSV unchanged, performs numeric coercion and
unknown-tolerant one-hot encoding, evaluates a held-out stratified split, and
saves the classifier and preprocessing objects together. The API accepts the
original Census column names through Pydantic aliases, including names such as
`education-num` and `marital-status`.

The API has `GET /` for a health/welcome response and `POST /predict` for
inference. The model artifact must be created before starting the service.
