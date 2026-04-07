---
title: Data Cleaning OpenEnv Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Data Cleaning OpenEnv Environment

## Overview

This repository contains a real-world OpenEnv benchmark for interactive tabular data cleaning. The agent operates on messy employee-style datasets and must resolve common data preparation issues step by step: missing values, duplicate rows, wrong dtypes, inconsistent categorical values, and derived feature creation.

The implementation uses plain Python data structures instead of pandas so it stays lightweight for the hackathon constraints, Docker validation, and Hugging Face Spaces deployment.

The repository now follows the standard OpenEnv layout closely:

```text
openenv-data-cleaning/
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── server/
│   ├── app.py
│   ├── environment.py
│   └── requirements.txt
└── outputs/
    ├── evals/
    └── logs/
```

## Environment Summary

- Domain: tabular data cleaning and preparation
- Mode: simulation environment with `reset()`, `step()`, and `state()`
- API: FastAPI on port `7860`
- Tasks: `basic_cleaning`, `moderate_cleaning`, `full_pipeline`
- Difficulty curve: easy -> medium -> hard

## Action Space

| Action | Target | Required params | Validation rules |
| --- | --- | --- | --- |
| `fill_missing` | Specific column | `{"strategy": "mean" \| "median" \| "zero" \| "mode" \| "unknown"}` | Numeric columns allow `mean`, `median`, `zero`; categorical columns allow `mode`, `unknown`. |
| `drop_duplicates` | `__all__` | `{}` | Only valid when duplicate rows are still present. |
| `convert_dtype` | Specific column | `{"target_dtype": "int" \| "float" \| "str" \| "bool"}` | Target dtype must match the task configuration and values must be convertible. |
| `normalize_category` | Categorical column | `{}` | Only valid when case-only category inconsistencies remain. |
| `create_feature` | Registered feature name | `{"feature_name": "<name>"}` | Feature must be required by the task and its source column must already be clean enough to use. |

Invalid actions leave the dataset unchanged, emit `{"error": "invalid_action"}` in `info`, consume a step, and return reward `-0.05`.

## Observation and State Space

Every `reset()`, `step()`, and `state()` call returns the same typed observation payload:

| Field | Type | Description |
| --- | --- | --- |
| `data_preview` | `list[dict[str, Any]]` | First five rows of the current dataset |
| `columns` | `list[ColumnInfo]` | Per-column dtype, null count, and unique count |
| `pending_issues` | `list[Issue]` | Remaining fixable issues |
| `resolved_issues` | `list[Issue]` | Issues already credited as solved |
| `action_history` | `list[dict[str, Any]]` | Previous actions with reward and optional error |
| `quality_score` | `float` | Current quality score in `[0.0, 1.0]` |
| `steps_remaining` | `int` | Remaining episode budget |
| `total_rows` | `int` | Current number of rows |
| `total_issues_at_start` | `int` | Issues detected immediately after `reset()` |

## Tasks

| Task | Difficulty | Rows | Main issue profile |
| --- | --- | --- | --- |
| `basic_cleaning` | Easy | 20 | Missing `age`, missing `salary` |
| `moderate_cleaning` | Medium | 50 | Missing `age`, missing `salary`, missing `years_exp`, duplicate rows, wrong `salary` dtype |
| `full_pipeline` | Hard | 100 | Missing values, duplicate rows, wrong `salary` and `rating` dtypes, inconsistent `city`, inconsistent `department`, required `age_group` feature |

The hardest task includes explicit dependency chains such as fixing missing salary values before dtype conversion and cleaning source columns before feature creation.

## Reward and Grading

Step reward:

```text
reward = (new_quality - old_quality) + ordering_bonus - 0.01
ordering_bonus = 0.05 if dependencies were already satisfied else 0.0
```

Dataset quality score combines:

- Completeness: 40%
- Uniqueness: 30%
- Consistency: 30%

Task grader:

```text
correctness = issues_fixed / total_issues
efficiency = max(0, 1 - steps_taken / (2 * total_issues))
penalty = wrong_actions * 0.05
score = 0.8 * correctness + 0.2 * efficiency - penalty
```

Grader scores are deterministic, clamped to `[0.0, 1.0]`, and rounded to two decimals.

## Setup

### Python and install

The project requires Python `3.10+`. Python `3.11` is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run local checks

```bash
python test_env.py
openenv validate .
```

### Run the FastAPI app

```bash
uv run server
```

Equivalent direct command:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run the baseline inference script

The hackathon evaluator expects these environment variables:

```bash
export HF_TOKEN=...
export API_BASE_URL=...
export MODEL_NAME=...
python inference.py
```

The script uses the OpenAI Python client and emits the required `[START]`, `[STEP]`, and `[END]` structured logs.

### Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

## API Surface

- `GET /`
- `GET /health`
- `GET /metadata`
- `GET /tasks`
- `GET /schema`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /mcp`

## Baseline Scores

Deterministic scripted benchmark from `test_env.py`:

- `basic_cleaning`: `0.90`
- `moderate_cleaning`: `0.90`
- `full_pipeline`: `0.90`

Model-based baseline from `inference.py`:

- `basic_cleaning`: `0.90`
- `moderate_cleaning`: `0.41`
- `full_pipeline`: `0.20`

These scores were produced on April 8, 2026 using `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct` through the configured Hugging Face router. The run completed and emitted the required structured logs, but the provider returned HTTP `402` after the early steps, so the medium and hard tasks were penalized by fallback `parse_error` actions. For a stronger final baseline, top up credits or switch `API_BASE_URL` / `MODEL_NAME` to a provider with available quota and rerun `python inference.py`.

## Deployment

### Hugging Face Spaces

Deploy this repo as a Docker Space tagged with OpenEnv. After deployment, verify:

- the Space root responds with HTTP `200`
- `POST /reset` works on the live Space
- `openenv validate <space-url>` passes runtime validation

Recommended deploy command:

```bash
openenv push --repo-id kaustubhg73/data-cleaning-openenv --exclude .openenv-upload-ignore
```

Space link:

- https://huggingface.co/spaces/kaustubhg73/data-cleaning-openenv
