from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import median
from typing import Any

from env.actions import FEATURE_REGISTRY, is_missing, validate_action
from env.models import Action, ColumnInfo, Issue, Observation
from env.quality import compute_quality_score
from env.rewards import compute_reward

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DataCleaningEnv:
    def __init__(self, task_name: str = "basic_cleaning"):
        self.task_name = task_name
        self.task_config: dict[str, Any] = {}
        self.dataset: list[dict[str, Any]] = []
        self.original_dataset: list[dict[str, Any]] = []
        self.issues: list[Issue] = []
        self.pending_issues: list[Issue] = []
        self.resolved_issues: list[Issue] = []
        self.action_history: list[dict[str, Any]] = []
        self.steps_remaining = 0
        self.max_steps = 0
        self.total_issues_at_start = 0
        self.quality_score = 0.0
        self.expected_dtypes: dict[str, str] = {}
        self.required_features: list[str] = []
        self._issue_id_map: dict[tuple[str, str], str] = {}

    def reset(self) -> Observation:
        config_path = DATA_DIR / f"{self.task_name}.json"
        with config_path.open("r", encoding="utf-8") as handle:
            self.task_config = json.load(handle)

        self.dataset = copy.deepcopy(self.task_config["dataset"])
        self.original_dataset = copy.deepcopy(self.dataset)
        self.expected_dtypes = dict(self.task_config["expected_dtypes"])
        self.required_features = list(self.task_config.get("required_features", []))
        self.action_history = []
        self.resolved_issues = []
        self.max_steps = int(self.task_config["max_steps"])
        self.steps_remaining = self.max_steps

        self._issue_id_map = {}
        detected = self._detect_issues(self.dataset)
        self.pending_issues = detected
        self.issues = list(detected)
        self.total_issues_at_start = len(detected)
        self.quality_score = compute_quality_score(
            self.dataset,
            self._build_column_infos(),
            self.total_issues_at_start,
        )
        return self.state()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if not self.dataset:
            self.reset()

        self.steps_remaining -= 1
        old_quality = self.quality_score
        columns = self._build_column_infos()
        action_valid, message, matched_issue, dependency_ok = validate_action(
            self.dataset,
            self.pending_issues,
            columns,
            self.expected_dtypes,
            action,
            self.resolved_issues,
        )

        info: dict[str, Any] = {}
        if not action_valid:
            reward = compute_reward(old_quality, old_quality, False, False)
            info = {"error": "invalid_action", "message": message}
            self.action_history.append(
                {
                    "action_type": action.action_type,
                    "column": action.column,
                    "params": action.params,
                    "reward": reward,
                    "error": message,
                }
            )
            observation = self.state()
            done = self.steps_remaining <= 0 or len(self.pending_issues) == 0
            return observation, reward, done, info

        self._apply_action(action)
        redetected = self._detect_issues(self.dataset)
        self.pending_issues = redetected
        self.issues = list(redetected)

        if matched_issue and not self._issue_present(redetected, matched_issue.issue_type, matched_issue.column):
            self.resolved_issues.append(matched_issue)

        self.quality_score = compute_quality_score(
            self.dataset,
            self._build_column_infos(),
            self.total_issues_at_start,
        )
        reward = compute_reward(old_quality, self.quality_score, True, dependency_ok)
        self.action_history.append(
            {
                "action_type": action.action_type,
                "column": action.column,
                "params": action.params,
                "reward": reward,
                "error": None,
            }
        )
        observation = self.state()
        done = self.steps_remaining <= 0 or len(self.pending_issues) == 0
        return observation, reward, done, info

    def state(self) -> Observation:
        return Observation(
            data_preview=copy.deepcopy(self.dataset[:5]),
            columns=self._build_column_infos(),
            pending_issues=copy.deepcopy(self.pending_issues),
            resolved_issues=copy.deepcopy(self.resolved_issues),
            action_history=copy.deepcopy(self.action_history),
            quality_score=self.quality_score,
            steps_remaining=self.steps_remaining,
            total_rows=len(self.dataset),
            total_issues_at_start=self.total_issues_at_start,
        )

    def _detect_issues(self, dataset: list[dict[str, Any]]) -> list[Issue]:
        if not dataset:
            return []

        raw_issues: list[dict[str, Any]] = []
        columns = list(self.expected_dtypes.keys())

        for column in columns:
            missing_count = sum(1 for row in dataset if is_missing(row.get(column)))
            if missing_count:
                raw_issues.append(
                    {
                        "issue_type": "missing",
                        "column": column,
                        "description": f"Column '{column}' has {missing_count} missing values that should be filled.",
                    }
                )

        if self._has_duplicates(dataset):
            raw_issues.append(
                {
                    "issue_type": "duplicate",
                    "column": "__all__",
                    "description": "Dataset contains duplicate rows that should be removed.",
                }
            )

        for column in columns:
            expected_dtype = self.expected_dtypes[column]
            actual_dtype = self._infer_runtime_dtype(dataset, column)
            if expected_dtype in {"int", "float", "bool"} and actual_dtype != expected_dtype:
                raw_issues.append(
                    {
                        "issue_type": "wrong_dtype",
                        "column": column,
                        "description": (
                            f"Column '{column}' should be '{expected_dtype}' but is currently represented as '{actual_dtype}'."
                        ),
                    }
                )

        for column in columns:
            if self.expected_dtypes[column] != "str":
                continue
            if self._has_inconsistent_categories(dataset, column):
                raw_issues.append(
                    {
                        "issue_type": "inconsistent_category",
                        "column": column,
                        "description": f"Column '{column}' has inconsistent categorical values that differ only by casing.",
                    }
                )

        for feature_name in self.required_features:
            if not all(feature_name in row for row in dataset):
                raw_issues.append(
                    {
                        "issue_type": "missing_feature",
                        "column": feature_name,
                        "description": f"Required feature '{feature_name}' has not been created yet.",
                    }
                )

        for raw_issue in raw_issues:
            signature = (raw_issue["issue_type"], raw_issue["column"])
            if signature not in self._issue_id_map:
                self._issue_id_map[signature] = f"issue_{len(self._issue_id_map) + 1:03d}"

        issues: list[Issue] = []
        signature_to_id = {signature: issue_id for signature, issue_id in self._issue_id_map.items()}

        for raw_issue in raw_issues:
            signature = (raw_issue["issue_type"], raw_issue["column"])
            depends_on: list[str] = []

            if raw_issue["issue_type"] == "wrong_dtype" and raw_issue["column"] in {"salary", "rating"}:
                missing_signature = ("missing", raw_issue["column"])
                if missing_signature in signature_to_id:
                    depends_on.append(signature_to_id[missing_signature])

            if raw_issue["issue_type"] == "missing_feature":
                feature_name = raw_issue["column"]
                source_column = FEATURE_REGISTRY[feature_name]["source"]
                for dependency_type in ("missing", "wrong_dtype"):
                    source_signature = (dependency_type, source_column)
                    if source_signature in signature_to_id:
                        depends_on.append(signature_to_id[source_signature])

            issues.append(
                Issue(
                    issue_id=signature_to_id[signature],
                    issue_type=raw_issue["issue_type"],
                    column=raw_issue["column"],
                    description=raw_issue["description"],
                    depends_on=depends_on,
                )
            )

        return issues

    def _build_column_infos(self) -> list[ColumnInfo]:
        if not self.dataset:
            return []

        infos: list[ColumnInfo] = []
        for column in self.dataset[0].keys():
            values = [row.get(column) for row in self.dataset]
            non_missing = [value for value in values if not is_missing(value)]
            infos.append(
                ColumnInfo(
                    name=column,
                    dtype=self._infer_runtime_dtype(self.dataset, column),
                    null_count=sum(1 for value in values if is_missing(value)),
                    unique_count=len({str(value) for value in non_missing}),
                )
            )
        return infos

    def _infer_runtime_dtype(self, dataset: list[dict[str, Any]], column: str) -> str:
        values = [row.get(column) for row in dataset if not is_missing(row.get(column))]
        if not values:
            return self.expected_dtypes.get(column, "str")
        if all(isinstance(value, bool) for value in values):
            return "bool"
        if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            return "int"
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return "float"
        return "str"

    def _has_duplicates(self, dataset: list[dict[str, Any]]) -> bool:
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for row in dataset:
            key = tuple(sorted(row.items()))
            if key in seen:
                return True
            seen.add(key)
        return False

    def _has_inconsistent_categories(self, dataset: list[dict[str, Any]], column: str) -> bool:
        groups: dict[str, set[str]] = {}
        for row in dataset:
            value = row.get(column)
            if is_missing(value):
                continue
            normalized = str(value).lower()
            groups.setdefault(normalized, set()).add(str(value))
        return any(len(forms) > 1 for forms in groups.values())

    def _issue_present(self, issues: list[Issue], issue_type: str, column: str) -> bool:
        return any(issue.issue_type == issue_type and issue.column == column for issue in issues)

    def _apply_action(self, action: Action) -> None:
        if action.action_type == "fill_missing":
            self._apply_fill_missing(action.column, action.params["strategy"])
        elif action.action_type == "drop_duplicates":
            unique_rows: list[dict[str, Any]] = []
            seen: set[tuple[tuple[str, Any], ...]] = set()
            for row in self.dataset:
                key = tuple(sorted(row.items()))
                if key in seen:
                    continue
                seen.add(key)
                unique_rows.append(row)
            self.dataset = unique_rows
        elif action.action_type == "convert_dtype":
            target_dtype = action.params["target_dtype"]
            for row in self.dataset:
                value = row.get(action.column)
                if is_missing(value):
                    row[action.column] = None
                else:
                    row[action.column] = self._convert_value(value, target_dtype)
        elif action.action_type == "normalize_category":
            self._apply_normalize_category(action.column)
        elif action.action_type == "create_feature":
            self._apply_create_feature(action.params["feature_name"])

    def _apply_fill_missing(self, column: str, strategy: str) -> None:
        expected_dtype = self.expected_dtypes.get(column, "str")
        valid_values = [row.get(column) for row in self.dataset if not is_missing(row.get(column))]

        if expected_dtype in {"int", "float"}:
            numeric_values = [self._convert_value(value, expected_dtype) for value in valid_values]
            if strategy == "mean":
                fill_value = sum(numeric_values) / len(numeric_values)
            elif strategy == "median":
                fill_value = median(numeric_values)
            else:
                fill_value = 0
            if expected_dtype == "int":
                fill_value = int(round(fill_value))
        else:
            if strategy == "mode":
                fill_value = self._pick_mode([str(value) for value in valid_values])
            else:
                fill_value = "unknown"

        for row in self.dataset:
            if is_missing(row.get(column)):
                row[column] = fill_value

    def _apply_normalize_category(self, column: str) -> None:
        groups: dict[str, dict[str, int]] = {}
        for row in self.dataset:
            value = row.get(column)
            if is_missing(value):
                continue
            surface = str(value)
            groups.setdefault(surface.lower(), {})
            groups[surface.lower()][surface] = groups[surface.lower()].get(surface, 0) + 1

        canonical: dict[str, str] = {}
        for lowered, counts in groups.items():
            canonical[lowered] = min(
                counts.items(),
                key=lambda item: (-item[1], item[0].lower(), 0 if item[0].islower() else 1, item[0]),
            )[0]

        for row in self.dataset:
            value = row.get(column)
            if is_missing(value):
                continue
            row[column] = canonical[str(value).lower()]

    def _apply_create_feature(self, feature_name: str) -> None:
        feature_config = FEATURE_REGISTRY[feature_name]
        source = feature_config["source"]
        bins = feature_config["bins"]
        labels = feature_config["labels"]

        for row in self.dataset:
            source_value = row.get(source)
            if is_missing(source_value):
                row[feature_name] = None
                continue

            numeric_value = float(source_value)
            assigned = None
            for index, label in enumerate(labels):
                lower = bins[index]
                upper = bins[index + 1]
                is_last = index == len(labels) - 1
                if (lower <= numeric_value < upper) or (is_last and lower <= numeric_value <= upper):
                    assigned = label
                    break
            row[feature_name] = assigned

    def _pick_mode(self, values: list[str]) -> str:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return min(
            counts.items(),
            key=lambda item: (-item[1], item[0].lower(), 0 if item[0].islower() else 1, item[0]),
        )[0]

    def _convert_value(self, value: Any, target_dtype: str) -> Any:
        if target_dtype == "int":
            return int(float(str(value)))
        if target_dtype == "float":
            return float(str(value))
        if target_dtype == "bool":
            normalized = str(value).strip().lower()
            return normalized in {"true", "1", "yes"}
        return str(value)
