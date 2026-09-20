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

Options
-------
--model       Model name/identifier passed to the provider.
--provider    LLM provider: 'openai', 'anthropic', or 'stub' (default: stub).
--runs-dir    Directory to save session JSON files (default: runs).
--budget      Hard time budget in seconds (default: 60).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


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

        calls = _sequence[step]

        # On submit step, populate features from accumulated observations
        if step == len(_sequence) - 1:
            obs = dict(_state["observations"])
            calls[0]["arguments"]["features"].update(obs)

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

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
            tool_calls = None

        if msg.tool_calls:
            _Resp.tool_calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                for tc in msg.tool_calls
            ]
        return _Resp()

    return chat


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

def _make_anthropic_llm(model_name: str) -> Any:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        sys.exit("[run_session] anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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
            tool_calls = None

        tcs = [b for b in response.content if b.type == "tool_use"]
        if tcs:
            _Resp.tool_calls = [
                {"id": tc.id, "name": tc.name, "arguments": tc.input}
                for tc in tcs
            ]
        return _Resp()

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
        choices=["stub", "openai", "anthropic"],
        help="LLM provider (default: stub).",
    )
    p.add_argument("--runs-dir", default="runs", help="Output directory for session JSON.")
    p.add_argument("--budget", type=float, default=60.0, help="Hard budget in seconds.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Override budget if custom value supplied
    if args.budget != 60.0:
        import src.harness as _harness_mod  # noqa: PLC0415
        _harness_mod.BUDGET_SEC = args.budget

    from src.harness import AcousticHarness  # noqa: PLC0415

    # Build LLM function
    if args.provider == "openai":
        llm_fn = _make_openai_llm(args.model)
    elif args.provider == "anthropic":
        llm_fn = _make_anthropic_llm(args.model)
    else:
        llm_fn = _make_stub_llm()

    harness = AcousticHarness(model_name=args.model, runs_dir=args.runs_dir)

    print(f"[run_session] Starting session – model={args.model}, provider={args.provider}, budget={args.budget}s")
    t0 = time.monotonic()
    result = harness.run_session(llm_fn)
    elapsed = time.monotonic() - t0

    print(f"\n[run_session] Session finished in {elapsed:.1f}s")
    print(f"  Termination : {result['termination_reason']}")
    print(f"  Features    : {json.dumps(result['final_features'], indent=2)}")

    # Save path is logged for convenience
    runs_dir = args.runs_dir
    candidates = sorted(
        (f for f in os.listdir(runs_dir) if f.startswith(args.model) and f.endswith(".json")),
        reverse=True,
    )
    if candidates:
        print(f"  Saved to    : {os.path.join(runs_dir, candidates[0])}")


if __name__ == "__main__":
    main()
