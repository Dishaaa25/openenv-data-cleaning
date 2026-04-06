from typing import Any

from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int


class Issue(BaseModel):
    issue_id: str
    issue_type: str
    column: str
    description: str
    depends_on: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    data_preview: list[dict[str, Any]]
    columns: list[ColumnInfo]
    pending_issues: list[Issue]
    resolved_issues: list[Issue]
    action_history: list[dict[str, Any]]
    quality_score: float
    steps_remaining: int
    total_rows: int
    total_issues_at_start: int


class Action(BaseModel):
    action_type: str
    column: str
    params: dict[str, str] = Field(default_factory=dict)


class Reward(BaseModel):
    value: float
