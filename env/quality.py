from __future__ import annotations

from typing import Any

from env.actions import is_missing


def _is_numeric_value(value: Any, dtype: str) -> bool:
    if is_missing(value):
        return False
    try:
        if dtype == "int":
            int(str(value))
        elif dtype == "float":
            float(str(value))
        else:
            return False
        return True
    except (TypeError, ValueError):
        return False


def _compute_consistency(dataset: list[dict], column_infos: list) -> float:
    if not dataset or not column_infos:
        return 1.0

    valid_checks = 0
    total_checks = 0

    for info in column_infos:
        values = [row.get(info.name) for row in dataset]
        if info.dtype in {"int", "float"}:
            for value in values:
                total_checks += 1
                if _is_numeric_value(value, info.dtype):
                    valid_checks += 1
        else:
            non_missing = [str(value) for value in values if not is_missing(value)]
            if not non_missing:
                continue
            lowered = {}
            for value in non_missing:
                lowered.setdefault(value.lower(), set()).add(value)
            has_inconsistency = any(len(forms) > 1 for forms in lowered.values())
            total_checks += 1
            if not has_inconsistency:
                valid_checks += 1

    return valid_checks / total_checks if total_checks else 1.0


def compute_quality_score(dataset: list[dict], column_infos: list, original_issues_count: int) -> float:
    if original_issues_count == 0:
        return 0.99

    total_cells = len(dataset) * len(dataset[0]) if dataset else 1
    missing_cells = sum(
        1 for row in dataset for value in row.values() if value is None or value == "" or value == "not_available"
    )
    completeness = 1.0 - (missing_cells / total_cells)

    total_rows = len(dataset)
    unique_rows = len(set(str(sorted(row.items())) for row in dataset))
    uniqueness = unique_rows / total_rows if total_rows > 0 else 1.0

    consistency = _compute_consistency(dataset, column_infos)

    score = 0.4 * completeness + 0.3 * uniqueness + 0.3 * consistency
    return round(max(0.01, min(0.99, score)), 4)
