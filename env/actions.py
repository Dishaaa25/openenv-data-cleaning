from __future__ import annotations

from typing import Any

from env.models import Action, ColumnInfo, Issue

ALLOWED_ACTIONS = {
    "fill_missing",
    "drop_duplicates",
    "convert_dtype",
    "normalize_category",
    "create_feature",
}

VALID_FILL_STRATEGIES = {
    "numeric": ["mean", "median", "zero"],
    "categorical": ["mode", "unknown"],
}

VALID_TARGET_DTYPES = {"int", "float", "str", "bool"}

FEATURE_REGISTRY = {
    "age_group": {
        "source": "age",
        "transform": "bin",
        "bins": [0, 18, 35, 50, 100],
        "labels": ["young", "adult", "middle", "senior"],
    },
    "salary_bracket": {
        "source": "salary",
        "transform": "bin",
        "bins": [0, 25000, 50000, 100000, float("inf")],
        "labels": ["low", "medium", "high", "very_high"],
    },
}

MISSING_SENTINELS = {None, "", "not_available"}


def is_missing(value: Any) -> bool:
    return value in MISSING_SENTINELS


def infer_column_family(expected_dtype: str) -> str:
    return "numeric" if expected_dtype in {"int", "float"} else "categorical"


def has_duplicates(dataset: list[dict[str, Any]]) -> bool:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for row in dataset:
        key = tuple(sorted(row.items()))
        if key in seen:
            return True
        seen.add(key)
    return False


def _get_column_info(column_infos: list[ColumnInfo], column: str) -> ColumnInfo | None:
    for info in column_infos:
        if info.name == column:
            return info
    return None


def _non_missing_values(dataset: list[dict[str, Any]], column: str) -> list[Any]:
    return [row.get(column) for row in dataset if not is_missing(row.get(column))]


def _is_convertible(value: Any, target_dtype: str) -> bool:
    if is_missing(value):
        return True
    try:
        if target_dtype == "int":
            if isinstance(value, bool):
                return True
            if isinstance(value, str) and value.strip() == "":
                return False
            int(str(value))
            return True
        if target_dtype == "float":
            float(str(value))
            return True
        if target_dtype == "bool":
            normalized = str(value).strip().lower()
            return normalized in {"true", "false", "1", "0", "yes", "no"}
        if target_dtype == "str":
            str(value)
            return True
    except (TypeError, ValueError):
        return False
    return False


def validate_action(
    dataset: list[dict[str, Any]],
    pending_issues: list[Issue],
    column_infos: list[ColumnInfo],
    expected_dtypes: dict[str, str],
    action: Action,
    resolved_issues: list[Issue],
) -> tuple[bool, str, Issue | None, bool]:
    if action.action_type not in ALLOWED_ACTIONS:
        return False, f"Unsupported action_type '{action.action_type}'", None, False

    issue_lookup = {(issue.issue_type, issue.column): issue for issue in pending_issues}
    column_info = _get_column_info(column_infos, action.column) if action.column != "__all__" else None
    resolved_ids = {issue.issue_id for issue in resolved_issues}

    matched_issue: Issue | None = None
    if action.action_type == "fill_missing":
        matched_issue = issue_lookup.get(("missing", action.column))
        if matched_issue is None:
            return False, f"Column '{action.column}' does not have a pending missing-value issue", None, False
        if column_info is None:
            return False, f"Unknown column '{action.column}'", None, False
        expected_dtype = expected_dtypes.get(action.column, column_info.dtype)
        family = infer_column_family(expected_dtype)
        strategy = action.params.get("strategy")
        if strategy not in VALID_FILL_STRATEGIES[family]:
            return False, f"Invalid fill strategy '{strategy}' for {family} column", None, False
        if not any(is_missing(row.get(action.column)) for row in dataset):
            return False, f"Column '{action.column}' has no missing values", None, False
    elif action.action_type == "drop_duplicates":
        matched_issue = issue_lookup.get(("duplicate", "__all__"))
        if action.column != "__all__":
            return False, "drop_duplicates must target column '__all__'", None, False
        if action.params:
            return False, "drop_duplicates does not accept params", None, False
        if matched_issue is None or not has_duplicates(dataset):
            return False, "Dataset does not have duplicate rows", None, False
    elif action.action_type == "convert_dtype":
        matched_issue = issue_lookup.get(("wrong_dtype", action.column))
        if matched_issue is None:
            return False, f"Column '{action.column}' does not have a pending wrong_dtype issue", None, False
        target_dtype = action.params.get("target_dtype")
        if target_dtype not in VALID_TARGET_DTYPES:
            return False, f"Invalid target dtype '{target_dtype}'", None, False
        if target_dtype != expected_dtypes.get(action.column):
            return False, f"Target dtype for '{action.column}' must be '{expected_dtypes.get(action.column)}'", None, False
        values = _non_missing_values(dataset, action.column)
        if any(not _is_convertible(value, target_dtype) for value in values):
            return False, f"Column '{action.column}' contains non-convertible values", None, False
        if any(str(value).strip().lower() == "not_available" for value in values):
            return False, f"Column '{action.column}' still contains not_available placeholders", None, False
    elif action.action_type == "normalize_category":
        matched_issue = issue_lookup.get(("inconsistent_category", action.column))
        if matched_issue is None:
            return False, f"Column '{action.column}' does not have a pending inconsistent_category issue", None, False
        if action.params:
            return False, "normalize_category does not accept params", None, False
        values = [row.get(action.column) for row in dataset if not is_missing(row.get(action.column))]
        lowered = [str(value).lower() for value in values]
        if len(lowered) == len(set(lowered)):
            return False, f"Column '{action.column}' has no categorical inconsistencies", None, False
    elif action.action_type == "create_feature":
        matched_issue = issue_lookup.get(("missing_feature", action.column))
        feature_name = action.params.get("feature_name")
        if matched_issue is None:
            return False, f"Column '{action.column}' does not have a pending missing_feature issue", None, False
        if feature_name not in FEATURE_REGISTRY:
            return False, f"Unknown feature '{feature_name}'", None, False
        if action.column != feature_name:
            return False, f"create_feature column must match feature name '{feature_name}'", None, False
        source_column = FEATURE_REGISTRY[feature_name]["source"]
        if source_column not in dataset[0]:
            return False, f"Source column '{source_column}' is missing", None, False
        source_dtype = expected_dtypes.get(source_column)
        if source_dtype not in {"int", "float"}:
            return False, f"Source column '{source_column}' must be numeric", None, False
        source_values = _non_missing_values(dataset, source_column)
        if any(not _is_convertible(value, source_dtype) for value in source_values):
            return False, f"Source column '{source_column}' is not clean enough to create the feature", None, False

    dependency_ok = True
    if matched_issue and matched_issue.depends_on:
        dependency_ok = all(dep_id in resolved_ids for dep_id in matched_issue.depends_on)
        if not dependency_ok:
            return False, f"Dependencies for issue '{matched_issue.issue_id}' are not resolved", matched_issue, False

    return True, "", matched_issue, dependency_ok
