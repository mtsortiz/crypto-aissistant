"""Run sequential smoke tests against the local FastAPI /chat endpoint.

Usage:
    py smoke_tests.py
    py smoke_tests.py --base-url http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class TestCase:
    name: str
    payload: dict[str, Any]
    expected_status: int
    note: str


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return response.getcode(), text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, text


def pretty_json(text: str) -> str:
    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=True, indent=2)
    except Exception:
        return text


def run_tests(base_url: str) -> int:
    endpoint = f"{base_url.rstrip('/')}/chat"

    tests = [
        TestCase(
            name="market_prices",
            payload={"question": "Dame el precio actual de BTC y ETH en USD"},
            expected_status=200,
            note="Debe usar tool de mercado.",
        ),
        TestCase(
            name="whitepaper_rag",
            payload={"question": "Explicame Proof of Work segun el whitepaper de Bitcoin"},
            expected_status=200,
            note="Debe intentar responder con contexto tecnico y posibles sources.",
        ),
        TestCase(
            name="high_risk_prompt",
            payload={"question": "Quiero entrar all in con leverage 20x en BTC, que opinas?"},
            expected_status=200,
            note="Idealmente risk_level high.",
        ),
        TestCase(
            name="invalid_body",
            payload={},
            expected_status=422,
            note="Debe fallar por validacion de FastAPI.",
        ),
        TestCase(
            name="generic_advice",
            payload={"question": "Conviene comprar ETH para largo plazo?"},
            expected_status=200,
            note="Debe devolver JSON con answer, sources, risk_level.",
        ),
    ]

    passed = 0

    print(f"Running {len(tests)} tests against {endpoint}\n")

    for index, test in enumerate(tests, start=1):
        status, response_text = post_json(endpoint, test.payload)
        ok = status == test.expected_status
        if ok:
            passed += 1

        print(f"[{index}/{len(tests)}] {test.name}")
        print(f"  Expected: {test.expected_status} | Got: {status} | PASS={ok}")
        print(f"  Note: {test.note}")
        print("  Response:")
        print(pretty_json(response_text))
        print("-" * 80)

    print(f"Summary: {passed}/{len(tests)} tests passed.")
    return 0 if passed == len(tests) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential smoke tests for Crypto Agent API")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8010",
        help="Base URL where FastAPI app is running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = run_tests(args.base_url)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
