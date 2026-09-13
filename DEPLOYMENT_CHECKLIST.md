# Streamlit Community Cloud deployment checklist

## 1. Replace the GitHub repository

Upload every file and folder from the ZIP to the repository root. The root should contain:

- `app.py`
- `ai_pipeline.py`
- `requirements.txt`
- `requirements-optional-boosters.txt`
- `packages.txt` — this file must be empty
- `.python-version`
- `runtime.txt`
- `.streamlit/config.toml`
- `vendor/`
- `notebooks/`
- `tests/`

Delete obsolete dependency files that are not part of the new ZIP.

## 2. Confirm the critical files

`requirements.txt` must begin with:

```text
--only-binary=:all:

streamlit==1.63.0
```

`.python-version` must contain:

```text
3.12
```

`runtime.txt` must contain:

```text
python-3.12
```

`packages.txt` must contain no package names.

## 3. Recreate the Community Cloud app

The existing deployment in the screenshot uses Python 3.14.7. Do not merely reboot it.

1. Record any Streamlit secrets and the current app URL.
2. Delete the old app from Community Cloud.
3. Click **Create app**.
4. Select the updated GitHub repository and branch.
5. Set **Main file path** to `app.py`.
6. Open **Advanced settings**.
7. Select **Python 3.12**.
8. Re-enter secrets if applicable.
9. Deploy.

## 4. Verify the build log

Expected:

```text
Using Python 3.12...
```

Incorrect:

```text
Using Python 3.14...
```

If the log still reports Python 3.14, the old deployment was rebooted instead of recreated, or Python 3.14 was selected again.

## 5. Verify the app

The app should open with:

- AI Swimming Coach heading
- Author: Jasper Ding
- Advisor: Dr. Qingyang Xiao
- Video uploader
- Analysis settings
- Deployment diagnostics panel

Run the computer-vision dependency check from the sidebar before uploading a large video.
