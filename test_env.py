import json
from pathlib import Path

from env.environment import DataCleaningEnv
from env.graders import DataCleaningGrader
from env.models import Action

ROOT = Path(__file__).resolve().parent


def assert_invalid_action_consumes_step() -> None:
    env = DataCleaningEnv("basic_cleaning")
    obs = env.reset()
    _, reward, _, info = env.step(
        Action(action_type="convert_dtype", column="age", params={"target_dtype": "int"})
    )
    assert reward == -0.05
    assert info["error"] == "invalid_action"
    assert env.steps_remaining == obs.steps_remaining - 1


def assert_dependency_gate() -> None:
    env = DataCleaningEnv("moderate_cleaning")
    env.reset()
    _, reward, _, info = env.step(
        Action(action_type="convert_dtype", column="salary", params={"target_dtype": "int"})
    )
    assert reward == -0.05
    assert info["error"] == "invalid_action"


def run_sequence(task_name: str, actions: list[Action], expected_issues: int) -> tuple[dict, float]:
    env = DataCleaningEnv(task_name)
    obs = env.reset()
    assert len(obs.pending_issues) == expected_issues, (task_name, len(obs.pending_issues), expected_issues)
    initial_quality = obs.quality_score

    for action in actions:
        obs, reward, done, info = env.step(action)
        assert "error" not in info, (task_name, action, info)
        if done:
            break

    assert obs.quality_score >= initial_quality
    final_state = obs.model_dump()
    config = json.loads((ROOT / "data" / f"{task_name}.json").read_text(encoding="utf-8"))
    score = DataCleaningGrader().grade(
        final_state,
        {
            "total_issues": expected_issues,
            "max_steps": config["max_steps"],
        },
    )
    return final_state, score


def main() -> None:
    assert_invalid_action_consumes_step()
    assert_dependency_gate()

    sequences = {
        "basic_cleaning": (
            [
                Action(action_type="fill_missing", column="age", params={"strategy": "mean"}),
                Action(action_type="fill_missing", column="salary", params={"strategy": "median"}),
            ],
            2,
        ),
        "moderate_cleaning": (
            [
                Action(action_type="fill_missing", column="age", params={"strategy": "mean"}),
                Action(action_type="fill_missing", column="years_exp", params={"strategy": "median"}),
                Action(action_type="fill_missing", column="salary", params={"strategy": "median"}),
                Action(action_type="convert_dtype", column="salary", params={"target_dtype": "int"}),
                Action(action_type="drop_duplicates", column="__all__", params={}),
            ],
            5,
        ),
        "full_pipeline": (
            [
                Action(action_type="fill_missing", column="age", params={"strategy": "mean"}),
                Action(action_type="fill_missing", column="years_exp", params={"strategy": "median"}),
                Action(action_type="fill_missing", column="rating", params={"strategy": "mean"}),
                Action(action_type="fill_missing", column="salary", params={"strategy": "median"}),
                Action(action_type="convert_dtype", column="salary", params={"target_dtype": "int"}),
                Action(action_type="convert_dtype", column="rating", params={"target_dtype": "float"}),
                Action(action_type="normalize_category", column="city", params={}),
                Action(action_type="normalize_category", column="department", params={}),
                Action(action_type="create_feature", column="age_group", params={"feature_name": "age_group"}),
                Action(action_type="drop_duplicates", column="__all__", params={}),
            ],
            10,
        ),
    }

    for task_name, (actions, expected_issues) in sequences.items():
        final_state, score = run_sequence(task_name, actions, expected_issues)
        pending = len(final_state["pending_issues"])
        resolved = len(final_state["resolved_issues"])
        print(
            f"{task_name}: pending={pending} resolved={resolved} "
            f"steps_remaining={final_state['steps_remaining']} grader_score={score}"
        )


if __name__ == "__main__":
    main()
