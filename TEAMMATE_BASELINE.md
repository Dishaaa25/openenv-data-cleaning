# Teammate Baseline Runbook

Use this if the current Hugging Face router quota is exhausted and you want to rerun the official baseline with a different token or provider.

## What is already done

- The OpenEnv environment is implemented and validated.
- The Hugging Face Space is live:
  - `https://huggingface.co/spaces/kaustubhg73/data-cleaning-openenv`
- Local validation, Docker validation, and live runtime validation already passed.

## What you need

Set these environment variables before running `inference.py`:

```bash
export HF_TOKEN=...
export API_BASE_URL=...
export MODEL_NAME=...
```

Important:

- `inference.py` uses the OpenAI Python client.
- In this repo, `HF_TOKEN` is the actual API key variable used by the client.
- A standard Hugging Face router configuration is:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="openai/gpt-oss-120b"
export HF_TOKEN="YOUR_HF_TOKEN"
```

## Local setup

From the repo root:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
```

## Run the baseline

```bash
python inference.py
```

The required log format is:

- `[START]`
- `[STEP]`
- `[END]`

Do not change the log format before submission.

## Expected follow-up after a successful run

Update the model-based baseline section in `README.md` with:

- the final scores for all three tasks
- the model name used
- a short note that the run completed successfully

## Optional validation checks

```bash
python test_env.py
openenv validate .
openenv validate --url https://kaustubhg73-data-cleaning-openenv.hf.space
```

## If you need to redeploy

Use the exclude file so the local `OpenEnv/` tutorial folder is not uploaded:

```bash
openenv push --repo-id kaustubhg73/data-cleaning-openenv --exclude .openenv-upload-ignore
```
