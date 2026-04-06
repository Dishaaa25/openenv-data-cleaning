# Data Cleaning OpenEnv Environment

## Overview

This project is a submission-ready reinforcement learning environment for interactive tabular data cleaning. It models a realistic pipeline where an agent receives a messy dataset, inspects detected issues, and fixes them step by step using a constrained action space. The benchmark focuses on practical data preparation work such as imputing missing values, removing duplicate rows, correcting dtypes, normalizing inconsistent categories, and creating derived features.

The environment is implemented with pure Python lists of dictionaries and avoids pandas entirely, which keeps runtime and memory usage small enough for the OpenEnv hackathon constraints and Hugging Face Spaces deployment target.

## Action Space

| Action | Target | Required params | Validation rules |
| --- | --- | --- | --- |
| `fill_missing` | A specific column | `{"strategy": "mean" \| "median" \| "zero" \| "mode" \| "unknown"}` | Column must have a pending missing-value issue. Numeric columns only allow `mean`, `median`, `zero`; categorical columns only allow `mode`, `unknown`. |
| `drop_duplicates` | `__all__` | `{}` | Dataset must contain duplicate rows. No params are accepted. |
| `convert_dtype` | A specific column | `{"target_dtype": "int" \| "float" \| "str" \| "bool"}` | Column must have a pending `wrong_dtype` issue, target dtype must match the configured expected dtype, and all non-null values must be convertible. |
| `normalize_category` | A categorical column | `{}` | Column must have a pending `inconsistent_category` issue. The environment normalizes case-insensitive variants deterministically. |
| `create_feature` | Feature name such as `age_group` | `{"feature_name": "<name>"}` | Feature must be registered, required by the task, and its source column must exist and be clean enough to use. |

Invalid actions return a reward of `-0.05`, leave the dataset unchanged, include `{"error": "invalid_action"}` in `info`, and still consume one step.

## Observation Space

Every call to `reset()`, `step()`, and `state()` returns an observation with these fields:

| Field | Type | Description |
| --- | --- | --- |
| `data_preview` | `list[dict[str, Any]]` | First 5 rows of the current dataset |
| `columns` | `list[ColumnInfo]` | Per-column dtype, null count, and unique count |
| `pending_issues` | `list[Issue]` | Remaining issues the agent can still fix |
| `resolved_issues` | `list[Issue]` | Issues explicitly credited as fixed |
| `action_history` | `list[dict[str, Any]]` | Prior actions with reward and optional error |
| `quality_score` | `float` | Current quality score in `[0.0, 1.0]` |
| `steps_remaining` | `int` | Steps left before termination |
| `total_rows` | `int` | Current row count |
| `total_issues_at_start` | `int` | Number of issues detected at reset time |

## Tasks

| Task | Difficulty | Rows | Issue profile |
| --- | --- | --- | --- |
| `basic_cleaning` | Easy | 20 | Missing `age`, missing `salary` |
| `moderate_cleaning` | Medium | 50 | Missing `age`, missing `salary` placeholders, missing `years_exp`, duplicate rows, wrong `salary` dtype |
| `full_pipeline` | Hard | 100 | Missing `age`, missing `salary` placeholders, missing `years_exp`, missing `rating`, duplicate rows, wrong `salary` dtype, wrong `rating` dtype, inconsistent `city`, inconsistent `department`, required feature creation |

The hard task intentionally includes dependency chains such as `fill_missing(salary)` before `convert_dtype(salary)` and `fill_missing(rating)` before `convert_dtype(rating)`.

## Reward Design

Rewards are computed as:

```text
reward = (new_quality - old_quality) + ordering_bonus - 0.01
ordering_bonus = 0.05 if dependencies were already satisfied else 0.0
```

If an action is invalid, reward is always `-0.05`.

Quality score combines:

- Completeness: 40%
- Uniqueness: 30%
- Consistency: 30%

## Grader Logic

The grader is deterministic and depends on the final environment state plus task metadata:

```text
correctness = issues_fixed / total_issues
efficiency = max(0, 1 - steps_taken / (2 * total_issues))
penalty = wrong_actions * 0.05
score = 0.8 * correctness + 0.2 * efficiency - penalty
```

The final score is clamped to `[0.0, 1.0]` and rounded to 2 decimals.

## Setup

### Local install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run manual environment checks

```bash
python3 test_env.py
```

### Run the FastAPI app

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run the baseline inference script

```bash
export API_KEY=...
python3 inference.py
```

### Docker

```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

## Baseline Scores

Manual deterministic action-sequence verification:

- `basic_cleaning`: grader score `0.90`
- `moderate_cleaning`: grader score `0.90`
- `full_pipeline`: grader score `0.90`

Model-based baseline with `inference.py`:

- Pending local API credentials and runtime execution

## Dependencies

- `pydantic>=2.0`
- `openai>=1.0`
- `fastapi`
- `uvicorn`
