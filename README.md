# Census Income Classifier

## Submission Repository

Platform: GitHub

Repository URL: https://github.com/ayolanre/nd0821-c3-starter-code-master

## Submission Layout

The submitted application is in `starter/`; Render uses this as its root directory. The nested `nd0821-c3-starter-code-master/` directory is the original course starter snapshot, not a second deliverable.

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* **Option 1: Using pip and venv (Recommended)**
    * Ensure you have Python 3.13 installed
    * Create virtual environment: `python3.13 -m venv .venv`
    * Activate environment: `source .venv/bin/activate` (On Windows: `.venv\Scripts\activate`)
    * Install dependencies: `pip install -r starter/requirements.txt`

* **Option 2: Using conda**
    * Download and install conda if you don't have it already.
    * conda create -n [envname] "python=3.13" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn pydantic httpx matplotlib seaborn -c conda-forge
    * Install git either through conda ("conda install git") or through your CLI, e.g. sudo apt-get git.

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Track data and model artifacts with DVC, push them to the configured remote, and commit the DVC metadata to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to use Python 3.13 (same version as development).
    * Note: Add flake8 to requirements.txt if you want to use it for linting: `pip install flake8`

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* The FastAPI service is deployed on Render at https://nd0821-c3-starter-code-master-vbpv.onrender.com.
* Render builds from the `starter/` root directory, pulls the model from the Google Drive DVC remote, and starts Uvicorn.
* Run `python request_example.py` from `starter/` with `API_URL` set to the live service URL to verify inference.
