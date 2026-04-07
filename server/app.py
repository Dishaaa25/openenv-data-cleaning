from __future__ import annotations

import argparse
from typing import Any, Literal

import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel

from models import Action, Observation

from .environment import DataCleaningEnv

TASKS = ["basic_cleaning", "moderate_cleaning", "full_pipeline"]
ENV_NAME = "data_cleaning_env"
ENV_DESCRIPTION = (
    "RL environment for interactive tabular data cleaning and preparation. "
    "Agents must fix missing values, duplicates, dtype issues, category inconsistencies, "
    "and derived-feature requirements."
)

app = FastAPI(title="Data Cleaning OpenEnv", version="1.0.0")
ENV = DataCleaningEnv()


class ResetRequest(BaseModel):
    task_name: Literal["basic_cleaning", "moderate_cleaning", "full_pipeline"] = "basic_cleaning"


def _metadata() -> dict[str, Any]:
    return {
        "name": ENV_NAME,
        "description": ENV_DESCRIPTION,
        "version": "1.0.0",
        "tasks": TASKS,
        "mode": "simulation",
    }


@app.get("/")
def root() -> dict[str, Any]:
    payload = _metadata()
    payload["status"] = "ok"
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    return _metadata()


@app.get("/tasks")
def list_tasks() -> dict[str, list[str]]:
    return {"tasks": TASKS}


@app.get("/schema")
def schema() -> dict[str, Any]:
    observation_schema = Observation.model_json_schema()
    return {
        "action": Action.model_json_schema(),
        "observation": observation_schema,
        "state": observation_schema,
    }


@app.post("/mcp")
def mcp(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "error": {
            "code": -32601,
            "message": "MCP methods are not implemented for this benchmark.",
        },
    }


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict[str, Any]:
    effective_request = request or ResetRequest()
    ENV.task_name = effective_request.task_name
    observation = ENV.reset()
    return observation.model_dump()


@app.post("/step")
def step(action: Action) -> dict[str, Any]:
    observation, reward, done, info = ENV.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    if not ENV.dataset:
        ENV.reset()
    return ENV.state().model_dump()


def main(host: str | None = None, port: int | None = None) -> None:
    if host is None or port is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=7860)
        args = parser.parse_args()
        host = args.host if host is None else host
        port = args.port if port is None else port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
