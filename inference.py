"""
STDOUT FORMAT (must match exactly):
[START] task=<task_name> env=data_cleaning_env model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os

from openai import OpenAI

from env.environment import DataCleaningEnv
from env.models import Action

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "data_cleaning_env"

TASKS = ["basic_cleaning", "moderate_cleaning", "full_pipeline"]

SYSTEM_PROMPT = """You are an AI agent performing data cleaning on a tabular dataset.

You will receive an observation containing:
- data_preview: first 5 rows of the current dataset
- columns: column info (name, dtype, null_count, unique_count)
- pending_issues: list of issues to fix (each has issue_id, issue_type, column, description, depends_on)
- resolved_issues: issues already fixed
- action_history: your previous actions
- quality_score: current data quality (0.0-1.0)
- steps_remaining: how many actions you have left

You must respond with EXACTLY one JSON object representing your action:
{
    "action_type": "<one of: fill_missing, drop_duplicates, convert_dtype, normalize_category, create_feature>",
    "column": "<target column name or __all__ for drop_duplicates>",
    "params": {<strategy-specific params>}
}

Rules:
- fill_missing: params must have "strategy" key. Use "mean"/"median"/"zero" for numeric columns, "mode"/"unknown" for categorical.
- drop_duplicates: column = "__all__", params = {}
- convert_dtype: params must have "target_dtype" key (one of: int, float, str, bool)
- normalize_category: params = {}
- create_feature: params must have "feature_name" key (e.g., "age_group")

IMPORTANT: Fix dependencies first! Check the "depends_on" field of each issue. For example, fill missing string values in a column BEFORE converting its dtype.

Respond with ONLY the JSON object. No explanation, no markdown, no code blocks."""


def parse_action(response_text: str) -> Action:
    text = response_text.strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) > 1 else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    parsed = json.loads(text)
    return Action(**parsed)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action_str, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True)


def run_task(task_name: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataCleaningEnv(task_name=task_name)
    obs = env.reset()
    log_start(task_name, BENCHMARK, MODEL_NAME)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards_list = []
    step_count = 0
    done = False
    max_possible_steps = obs.steps_remaining

    while not done and step_count < max_possible_steps:
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
        messages.append(
            {
                "role": "user",
                "content": f"Current observation:\n{json.dumps(obs_dict, indent=2, default=str)}\n\nChoose your next action.",
            }
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=200,
            )
            response_text = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})

            action = parse_action(response_text)
            obs, reward, done, info = env.step(action)
            step_count += 1
            last_error = info.get("error")
            rewards_list.append(reward)

            action_str = f"{action.action_type}({action.column})"
            log_step(step_count, action_str, reward, done, last_error)

        except Exception as exc:
            step_count += 1
            rewards_list.append(-0.05)
            log_step(step_count, "parse_error", -0.05, False, str(exc))
            messages.append(
                {
                    "role": "user",
                    "content": f"Your response could not be parsed. Error: {str(exc)}. Respond with ONLY a valid JSON action object.",
                }
            )
            if step_count >= max_possible_steps:
                break

    success = hasattr(obs, "pending_issues") and len(obs.pending_issues) == 0
    log_end(success, step_count, rewards_list)


def main():
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()
