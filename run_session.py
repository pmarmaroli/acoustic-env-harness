#!/usr/bin/env python3
"""run_session.py – CLI entry point for acoustic-env-harness.

Usage examples
--------------
# Dry-run with the built-in stub LLM (no API key required):
    python run_session.py --model stub

# OpenAI GPT-4o (requires OPENAI_API_KEY env var):
    python run_session.py --model gpt-4o --provider openai

# Anthropic Claude 3.5 Sonnet (requires ANTHROPIC_API_KEY env var):
    python run_session.py --model claude-3-5-sonnet-20241022 --provider anthropic

# Ollama local model (requires a running Ollama server):
    python run_session.py --model mistral --provider ollama

Options
-------
--model       Model name/identifier passed to the provider.
--provider    LLM provider: 'openai', 'anthropic', 'ollama', or 'stub' (default: stub).
--runs-dir    Directory to save session JSON files (default: runs).
--budget      Hard time budget in seconds (default: 60).
--jev-url     Base URL for the Jev classifier API (default: http://localhost:8000).
--ollama-url  Base URL for the Ollama API (default: http://localhost:11434).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any
from urllib.parse import urlparse

import httpx


# ---------------------------------------------------------------------------
# Stub LLM – deterministic finite sequence of tool calls, no API needed
# ---------------------------------------------------------------------------

class _StubResponse:
    """Minimal response object mimicking an LLM tool-call response."""

    def __init__(self, tool_calls: list[dict]) -> None:
        self.tool_calls = tool_calls

def _make_stub_llm() -> Any:
    """Return a stub LLM function that cycles through a fixed call sequence."""
    _sequence = [
        [{"id": "tc1", "name": "listen_ambient", "arguments": {"duration_sec": 3.0}}],
        [{"id": "tc2", "name": "measure_ir", "arguments": {"duration_sec": 2.0}}],
        [
            {
                "id": "tc3",
                "name": "submit_jev_features",
                "arguments": {
                    "features": {
                        "t60_est_sec": None,       # filled from previous observations
                        "rms_db": None,
                        "low_frequency_energy_ratio": None,
                        "crest_factor": None,
                        "transients_detected": None,
                        "dominant_stationary_freq_hz": None,
                        "comb_filter_notch_hz": None,
                        "snr_peak_db": None,
                        "predicted_env": "office",
                    }
                },
            }
        ],
    ]
    _state: dict[str, Any] = {"step": 0, "observations": {}}

    def stub_llm(history: list[dict]) -> _StubResponse:
        # Merge any tool results back into observations
        for msg in history:
            if msg.get("role") == "tool":
                try:
                    obs = json.loads(msg["content"])
                    _state["observations"].update(obs)
                except (json.JSONDecodeError, TypeError):
                    pass

        step = _state["step"]
        if step >= len(_sequence):
            # Should not happen in normal flow; submit empty features as fallback
            return _StubResponse([{
                "id": "fallback",
                "name": "submit_jev_features",
                "arguments": {"features": _state["observations"]},
            }])

        calls = copy.deepcopy(_sequence[step])

        # On submit step, populate features from accumulated observations
        if step == len(_sequence) - 1:
            calls[0]["arguments"]["features"].update(_state["observations"])

        _state["step"] += 1
        return _StubResponse(calls)

    return stub_llm


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

def _make_openai_llm(model_name: str) -> Any:
    try:
        import openai  # noqa: PLC0415
    except ImportError:
        sys.exit("[run_session] openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[run_session] OPENAI_API_KEY environment variable is not set.")
    client = openai.OpenAI(api_key=api_key)

    from src.harness import TOOL_DEFINITIONS  # noqa: PLC0415

    tools = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in TOOL_DEFINITIONS
    ]

    def chat(history: list[dict]) -> Any:
        response = client.chat.completions.create(
            model=model_name,
            messages=history,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        class _Resp:
            def __init__(self) -> None:
                self.tool_calls = None

        resp = _Resp()
        if msg.tool_calls:
            resp.tool_calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                for tc in msg.tool_calls
            ]
        return resp

    return chat


# ---------------------------------------------------------------------------
# Ollama provider (OpenAI-compatible chat completions API)
# ---------------------------------------------------------------------------

def _make_ollama_llm(model_name: str, base_url: str) -> Any:
    from src.harness import TOOL_DEFINITIONS  # noqa: PLC0415

    tools = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in TOOL_DEFINITIONS
    ]
    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    def chat(history: list[dict]) -> Any:
        try:
            response = httpx.post(
                api_url,
                json={
                    "model": model_name,
                    "messages": history,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False,
                },
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.ConnectError:
            sys.exit(
                f"[run_session] Could not connect to Ollama at {base_url}. "
                "Start Ollama first or override --ollama-url."
            )
        except httpx.TimeoutException:
            sys.exit(f"[run_session] Timed out while waiting for Ollama at {api_url}.")
        except httpx.HTTPStatusError as exc:
            sys.exit(
                f"[run_session] Ollama request failed with HTTP {exc.response.status_code}: "
                f"{exc.response.text}"
            )
        except httpx.HTTPError as exc:
            sys.exit(f"[run_session] Ollama request failed: {exc}")

        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
            parsed_tool_calls = [
                {
                    "id": tc.get("id", tc["function"]["name"]),
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], str)
                    else tc["function"]["arguments"],
                }
                for tc in (message.get("tool_calls") or [])
            ]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            sys.exit(
                f"[run_session] Ollama returned an unexpected response format from {api_url}."
            )

        class _Resp:
            def __init__(self) -> None:
                self.tool_calls = None

        resp = _Resp()
        if parsed_tool_calls:
            resp.tool_calls = parsed_tool_calls
        return resp

    return chat


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

def _make_anthropic_llm(model_name: str) -> Any:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        sys.exit("[run_session] anthropic package not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("[run_session] ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    from src.harness import TOOL_DEFINITIONS  # noqa: PLC0415

    tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in TOOL_DEFINITIONS
    ]

    def chat(history: list[dict]) -> Any:
        # Anthropic separates system from the message list
        system = ""
        messages = []
        for m in history:
            if m["role"] == "system":
                system = m["content"]
            else:
                messages.append(m)

        response = client.messages.create(
            model=model_name,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools,
        )

        class _Resp:
            def __init__(self) -> None:
                self.tool_calls = None

        resp = _Resp()
        tcs = [b for b in response.content if b.type == "tool_use"]
        if tcs:
            resp.tool_calls = [
                {"id": tc.id, "name": tc.name, "arguments": tc.input}
                for tc in tcs
            ]
        return resp

    return chat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a 60-second acoustic exploration session with an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="stub", help="Model name (default: stub).")
    p.add_argument(
        "--provider",
        default="stub",
        choices=["stub", "openai", "anthropic", "ollama"],
        help="LLM provider (default: stub).",
    )
    p.add_argument("--runs-dir", default="runs", help="Output directory for session JSON.")
    p.add_argument("--budget", type=float, default=60.0, help="Hard budget in seconds.")
    p.add_argument(
        "--jev-url",
        default="http://localhost:8000",
        help="Base URL for the Jev classifier API (default: http://localhost:8000).",
    )
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for the Ollama API (default: http://localhost:11434).",
    )
    return p


def _validate_http_url(name: str, value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        sys.exit(f"[run_session] {name} must be a valid http(s) URL, got: {value!r}")
    return value.rstrip("/")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from src.harness import AcousticHarness  # noqa: PLC0415
    from src.jev_client import JevClient  # noqa: PLC0415

    jev_url = _validate_http_url("--jev-url", args.jev_url)
    ollama_url: str | None = None
    if args.provider == "ollama":
        ollama_url = _validate_http_url("--ollama-url", args.ollama_url)

    # Build LLM function
    if args.provider == "openai":
        llm_fn = _make_openai_llm(args.model)
    elif args.provider == "anthropic":
        llm_fn = _make_anthropic_llm(args.model)
    elif args.provider == "ollama":
        if ollama_url is None:
            sys.exit("[run_session] --ollama-url validation did not run as expected.")
        llm_fn = _make_ollama_llm(args.model, ollama_url)
    else:
        llm_fn = _make_stub_llm()

    harness = AcousticHarness(
        model_name=args.model,
        runs_dir=args.runs_dir,
        budget_sec=args.budget,
        jev_client=JevClient(base_url=jev_url),
    )

    print(
        "[run_session] Starting session – "
        f"model={args.model}, provider={args.provider}, budget={args.budget}s, jev_url={jev_url}"
    )
    t0 = time.monotonic()
    result = harness.run_session(llm_fn)
    elapsed = time.monotonic() - t0

    print(f"\n[run_session] Session finished in {elapsed:.1f}s")
    print(f"  Termination : {result['termination_reason']}")
    print(f"  Features    : {json.dumps(result['final_features'], indent=2)}")
    print(f"  Jev         : {json.dumps(result['jev_response'], indent=2)}")

    # Save path is logged for convenience
    runs_dir = args.runs_dir
    candidates: list[str] = []
    if os.path.isdir(runs_dir):
        candidates = sorted(
            (f for f in os.listdir(runs_dir) if f.startswith(args.model) and f.endswith(".json")),
            reverse=True,
        )
    if candidates:
        print(f"  Saved to    : {os.path.join(runs_dir, candidates[0])}")


if __name__ == "__main__":
    main()
