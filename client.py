"""Thin client helpers for local development and OpenEnv packaging."""

from typing import Any

import httpx


class DataCleaningEnvClient:
    """Minimal HTTP client for smoke-testing the environment locally."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def reset(self, task_name: str = "basic_cleaning") -> dict[str, Any]:
        response = self._client.post("/reset", json={"task_name": task_name})
        response.raise_for_status()
        return response.json()

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post("/step", json=payload)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        response = self._client.get("/state")
        response.raise_for_status()
        return response.json()


__all__ = ["DataCleaningEnvClient"]
