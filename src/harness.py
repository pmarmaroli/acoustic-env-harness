"""harness.py – Agent orchestration with a hard 60-second budget.

Exposes three tools to the LLM:
  • measure_ir        – active IR measurement
  • listen_ambient    – passive ambient recording
  • submit_jev_features – terminal action: submit feature vector

The harness injects ``remaining_budget_sec`` into every tool return so the
model can plan its actions within the time budget.  Full execution traces are
saved under ``runs/<model_name>_<unix_ts>.json``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.audio_engine import AudioEngine
from src.jev_client import JevClient, JevClientError

BUDGET_SEC: float = 60.0

SYSTEM_PROMPT = """You are an autonomous acoustic investigation agent deployed on a physical machine.

OBJECTIVE:
Extract and submit a feature vector that allows a statistical classifier to
discriminate the current acoustic environment among 4 classes:
  1. car      – enclosed cabin, strong continuous low-frequency noise, near-zero T60
  2. bathroom – hard reflective surfaces, high and bright T60
  3. outdoor  – free field, no late reverberation, wind/air noise
  4. office   – furnished space, moderate T60, comb-filter notch from desk reflection

OPERATIONAL CONSTRAINTS:
- You have a strict budget of 60 REAL SECONDS.
- Each audio tool call consumes physical real time.
- The time remaining will always be reported in each function return as `remaining_budget_sec`.
- As soon as your observations are sufficient, OR if time is running low,
  you MUST call `submit_jev_features(features)`.
- Exceeding 60 seconds without a validated submission counts as a FAILURE (timeout).
"""

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "measure_ir",
        "description": (
            "Emit a windowed logarithmic sine sweep and compute the impulse response "
            "(T60, comb-filter notch, SNR). Uses speakers + microphone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "f_min": {"type": "number", "default": 150.0, "description": "Lower frequency (Hz)."},
                "f_max": {"type": "number", "default": 14000.0, "description": "Upper frequency (Hz)."},
                "duration_sec": {"type": "number", "default": 2.0, "description": "Sweep duration (s)."},
            },
        },
    },
    {
        "name": "listen_ambient",
        "description": (
            "Passively record background noise without emitting any sound. "
            "Returns PSD peak, RMS level, transient detection and low-frequency ratio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "duration_sec": {"type": "number", "default": 4.0, "description": "Recording duration (s)."},
            },
        },
    },
    {
        "name": "submit_jev_features",
        "description": (
            "Terminal action: submit the acoustic feature vector to the Jev classifier "
            "and end the session. Must be called before the 60-second budget expires."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Free dictionary of numeric and categorical acoustic features.",
                }
            },
            "required": ["features"],
        },
    },
]


class SessionResult(BaseModel):
    """Validated session result payload."""

    model_name: str
    timestamp: str
    total_elapsed_sec: float
    termination_reason: str
    final_features: dict[str, Any] | None = Field(default=None)
    jev_response: dict[str, Any] | None = Field(default=None)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class AcousticHarness:
    """Orchestrate a bounded 60-second acoustic exploration session for one LLM."""

    def __init__(
        self,
        model_name: str,
        runs_dir: str = "runs",
        budget_sec: float = BUDGET_SEC,
        jev_client: JevClient | None = None,
    ) -> None:
        self.model_name = model_name
        self.runs_dir = runs_dir
        self.budget_sec = budget_sec
        self.audio = AudioEngine()
        self.jev_client = jev_client or JevClient()
        os.makedirs(self.runs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_session(
        self,
        llm_chat_fn: Callable[[list[dict[str, Any]]], Any],
    ) -> dict:
        """Run the 60-second session for one model.

        Parameters
        ----------
        llm_chat_fn : callable
            A function that accepts the conversation history (list of message
            dicts) and returns an object with a ``tool_calls`` attribute
            (list of ``{"id": str, "name": str, "arguments": dict}`` dicts).

        Returns
        -------
        dict
            Full session result payload (also saved to disk).
        """
        t_start = time.monotonic()
        history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        trace: list[dict[str, Any]] = []

        final_features: dict | None = None
        jev_response: dict[str, Any] | None = None
        termination_reason = "in_progress"

        while True:
            remaining = max(0.0, self.budget_sec - (time.monotonic() - t_start))
            if remaining <= 0.5:
                termination_reason = "timeout"
                break

            # --- LLM inference ---
            try:
                response = llm_chat_fn(history)
            except Exception as exc:  # noqa: BLE001
                termination_reason = "llm_error"
                trace.append({"event": "llm_error", "detail": str(exc)})
                break

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                termination_reason = "no_action"
                break

            # Append the assistant message with ALL tool calls at once (required by OpenAI/Anthropic)
            history.append({"role": "assistant", "tool_calls": tool_calls})

            session_done = False
            tool_results: list[dict] = []

            for tc in tool_calls:
                tool_name: str = tc.get("name", "")
                tool_args: dict = tc.get("arguments", {})
                tc_id: str = tc.get("id", tool_name)

                if tool_name == "submit_jev_features":
                    # Terminal tool – record a synthetic tool result to keep history consistent,
                    # then end the session immediately.
                    final_features = tool_args.get("features", {})
                    try:
                        jev_response = self.submit_jev_features(final_features)
                        termination_reason = "completed"
                        submit_result = {
                            "status": "accepted",
                            "jev_response": jev_response,
                            "elapsed_sec": round(time.monotonic() - t_start, 2),
                        }
                    except JevClientError as exc:
                        termination_reason = "jev_error"
                        submit_result = {
                            "status": "error",
                            "error": str(exc),
                            "elapsed_sec": round(time.monotonic() - t_start, 2),
                        }
                    trace.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": submit_result,
                        "elapsed_sec": submit_result["elapsed_sec"],
                    })
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(submit_result),
                    })
                    session_done = True
                    break

                # --- Dispatch audio tools ---
                remaining_before = max(0.0, self.budget_sec - (time.monotonic() - t_start))

                if tool_name == "measure_ir":
                    duration = min(float(tool_args.get("duration_sec", 2.0)), remaining_before)
                    res = self.audio.measure_ir(
                        f_min=float(tool_args.get("f_min", 150.0)),
                        f_max=float(tool_args.get("f_max", 14000.0)),
                        duration_sec=duration,
                    )
                elif tool_name == "listen_ambient":
                    duration = min(float(tool_args.get("duration_sec", 4.0)), remaining_before)
                    res = self.audio.listen_ambient(duration_sec=duration)
                else:
                    res = {"error": f"Unknown tool: {tool_name}"}

                # Inject remaining budget after the call completes
                remaining_after = max(0.0, self.budget_sec - (time.monotonic() - t_start))
                res["remaining_budget_sec"] = round(remaining_after, 2)

                trace.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": res,
                    "elapsed_sec": round(time.monotonic() - t_start, 2),
                })

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": json.dumps(res),
                })

            # Append all tool result messages after processing the full batch
            history.extend(tool_results)

            if session_done:
                break

        # --- Finalise ---
        total_elapsed = round(time.monotonic() - t_start, 2)
        if total_elapsed >= self.budget_sec and termination_reason != "completed":
            termination_reason = "timeout"

        result = SessionResult(
            model_name=self.model_name,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            total_elapsed_sec=total_elapsed,
            termination_reason=termination_reason,
            final_features=final_features,
            jev_response=jev_response,
            trace=trace,
        )

        self._save(result)
        return result.model_dump()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, result: SessionResult) -> None:
        filename = f"{self.model_name}_{int(time.time())}.json"
        filepath = os.path.join(self.runs_dir, filename)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(result.model_dump(), fh, indent=2, ensure_ascii=False)

    def submit_jev_features(self, features: dict[str, Any]) -> dict[str, Any]:
        """Submit the final acoustic feature vector to the Jev API."""
        return self.jev_client.submit_features(features)
