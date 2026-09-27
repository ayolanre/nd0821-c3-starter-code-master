Render is configured to install DVC with Google Drive support and pull the model before starting the API. The credential JSON is stored in Render as the secret file `gdrive_credentials.json`, mounted at `/etc/secrets/gdrive_credentials.json`.

The Render start command validates that file, supplies it to DVC through `GDRIVE_CREDENTIALS_DATA`, runs `python -m dvc pull -r gdrive`, and starts Uvicorn only if the pull succeeds. Do not add the credential JSON or cached token to the repository.
