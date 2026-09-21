"""Typed HTTP client for the Jev supervised acoustic classifier."""

from __future__ import annotations

from typing import Any, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError


class JevClientError(RuntimeError):
    """Raised when the Jev API cannot be reached or returns invalid data."""


class JevFeaturesRequest(BaseModel):
    """Validated request payload sent to the Jev API."""

    features: dict[str, Any]


class JevPredictionResponse(BaseModel):
    """Validated response payload returned by the Jev API."""

    model_config = ConfigDict(extra="allow")

    prediction: str
    confidence: float | None = None
    class_id: int | None = None


class JevClient:
    """Small, configurable client for Jev's HTTP classifier API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        endpoint_paths: Sequence[str] | None = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint_paths = tuple(endpoint_paths or ("/classify", "/predict"))
        self.timeout_sec = timeout_sec

    def submit_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """POST a feature dictionary to the Jev API and return its prediction."""
        payload = JevFeaturesRequest(features=features).model_dump()
        attempted_paths: list[str] = []
        last_error: str | None = None

        with httpx.Client(base_url=self.base_url, timeout=self.timeout_sec) as client:
            for endpoint_path in self.endpoint_paths:
                attempted_paths.append(endpoint_path)
                try:
                    response = client.post(endpoint_path, json=payload)
                    response.raise_for_status()
                    body = response.json()
                    return JevPredictionResponse.model_validate(body).model_dump()
                except httpx.ConnectError as exc:
                    last_error = (
                        f"Could not connect to Jev API at {self.base_url}. "
                        f"Verify that the service is running and that --jev-url is correct. ({exc})"
                    )
                    continue
                except httpx.TimeoutException as exc:
                    raise JevClientError(
                        f"Timed out while contacting Jev API at {self.base_url}{endpoint_path}."
                    ) from exc
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        last_error = (
                            f"{self.base_url}{endpoint_path} returned 404 Not Found"
                        )
                        continue
                    raise JevClientError(
                        f"Jev API request to {self.base_url}{endpoint_path} failed with "
                        f"HTTP {exc.response.status_code}: {exc.response.text}"
                    ) from exc
                except ValueError as exc:
                    raise JevClientError(
                        f"Jev API at {self.base_url}{endpoint_path} returned invalid JSON."
                    ) from exc
                except ValidationError as exc:
                    raise JevClientError(
                        f"Jev API response from {self.base_url}{endpoint_path} did not match "
                        f"the expected schema: {exc}"
                    ) from exc

        detail = last_error or "no usable endpoint responded"
        raise JevClientError(
            f"Unable to submit features to Jev API. Tried endpoints {attempted_paths}: {detail}."
        )
