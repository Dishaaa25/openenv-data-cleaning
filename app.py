from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from env.environment import DataCleaningEnv
from env.models import Action

TASKS = ["basic_cleaning", "moderate_cleaning", "full_pipeline"]

app = FastAPI(title="Data Cleaning OpenEnv")
ENV = DataCleaningEnv()


class ResetRequest(BaseModel):
    task_name: Literal["basic_cleaning", "moderate_cleaning", "full_pipeline"] = "basic_cleaning"


@app.get("/")
def root():
    return {
        "name": "data_cleaning_env",
        "status": "ok",
        "tasks": TASKS,
    }


@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS}


@app.post("/reset")
def reset(request: ResetRequest):
    ENV.task_name = request.task_name
    observation = ENV.reset()
    return observation.model_dump()


@app.post("/step")
def step(action: Action):
    observation, reward, done, info = ENV.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return ENV.state().model_dump()
