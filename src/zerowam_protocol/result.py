"""The result of one evaluated unit: what every benchmark writes and the competition core reads.

One JSON file, `result.json`, written even when the unit ended badly - a run that produced no file
is a harness failure, not a score. `outcome` is the only field scoring reads:

- `success`: the benchmark's own success check passed.
- `failure`: it did not, or the model's own side failed (an error reply, a malformed action, a
  timeout). It counts against that submission.
- `void`: nothing was learned about the model. `void_cause` says whose fault it was: `harness` for
  a simulator crash, a scene that did not restore, a GPU that went away; `runtime` for the served
  model process. A harness void is dropped for every submission, so all of them stay compared on
  the same units.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["OUTCOMES", "RESULT_JSON", "RESULT_VERSION", "VOID_CAUSES", "read", "write"]

#: Bumped whenever a required field changes.
RESULT_VERSION = 1

RESULT_JSON = "result.json"

OUTCOMES = ("success", "failure", "void")
VOID_CAUSES = ("harness", "runtime")

REQUIRED_KEYS = ("result_version", "unit_id", "demo_sha256", "outcome")


def write(
    out_dir: str | Path,
    *,
    unit_id: str,
    demo_sha256: str,
    outcome: str,
    void_cause: str | None = None,
    steps: int | None = None,
    step_limit: int | None = None,
    fingerprint_ok: bool | None = None,
    policy_calls: int | None = None,
    timing: Mapping[str, float] | None = None,
    error: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `result.json` and return it. A void needs a cause; anything else must not carry one."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome {outcome!r} is not one of {', '.join(OUTCOMES)}")
    if outcome == "void" and void_cause not in VOID_CAUSES:
        raise ValueError(f"a void needs void_cause in {', '.join(VOID_CAUSES)}, not {void_cause!r}")
    if outcome != "void" and void_cause is not None:
        raise ValueError(f"{outcome} must not carry void_cause {void_cause!r}")
    record: dict[str, Any] = {
        "result_version": RESULT_VERSION,
        "unit_id": unit_id,
        "demo_sha256": demo_sha256,
        "outcome": outcome,
        "void_cause": void_cause,
        "steps": steps,
        "step_limit": step_limit,
        "fingerprint_ok": fingerprint_ok,
        "policy_calls": policy_calls,
        "timing": dict(timing or {}),
        "error": error,
    }
    record.update(dict(extra or {}))
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / RESULT_JSON).write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record


def read(out_dir: str | Path) -> dict[str, Any]:
    """The `result.json` of a run directory, with its required fields checked."""
    path = Path(out_dir) / RESULT_JSON
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{path}: no such file") from None
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    if not isinstance(record, dict):
        raise ValueError(f"{path}: result is not a JSON object")
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"{path}: result is missing {', '.join(missing)}")
    if record["result_version"] != RESULT_VERSION:
        raise ValueError(f"{path}: result version {record['result_version']!r}")
    if record["outcome"] not in OUTCOMES:
        raise ValueError(f"{path}: outcome {record['outcome']!r}")
    return record
