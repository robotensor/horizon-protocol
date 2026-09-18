"""The result of one evaluated unit: what every benchmark writes and the competition core reads.

One JSON file, `result.json`, written even when the unit ended badly - a run that produced no file
is a harness failure, not a score. `outcome` is the only field scoring reads, classified by
decision Q6:

- `success`: the benchmark's own success check passed.
- `failure`: it did not, or the model's own side failed (an error reply, a malformed action, a
  timeout). It counts against that submission.
- `void`: nothing was learned about the model. `void_cause` says whose fault it was: `harness` for
  a simulator crash, a scene that did not restore, a GPU that went away; `runtime` for the served
  model process. A harness void is dropped for every submission, so all of them stay compared on
  the same units.

**Result version 2.** Every field below is in every result - `null` where it does not apply - and
`read` requires every one; `write` sets them all from its own arguments, and `extra` may add other
keys but never one of these (`REQUIRED_KEYS`), so nothing can overwrite an outcome:

- `result_version`: `RESULT_VERSION`; `read` refuses any other.
- `unit_id`: the unit's id, as its bundle's `demo.json` records it; non-empty.
- `demo_sha256`: the whole-bundle digest, `bundle.digest(bundle_dir)`: the sha256 of the bundle's
  `demo.json`, which hashes every other file of the unit, `private/` included. Not the frames'
  hash: a result names the exact unit, scene included, it was produced on.
- `outcome` and `void_cause`: a `void` has a cause in `VOID_CAUSES`, a `success` or `failure` has
  none (Q6). `read` applies the same rule.
- `task_config`, `task_config_sha256`, `fork_commit`: which task config the fork evaluated under,
  its sha256, and the fork's commit, so every result records what produced it.
- `steps`, `step_limit`, `policy_calls` (integers) and `fingerprint_ok` (a boolean), each `null`
  when the unit ended before it was known.
- `timing`: exactly `TIMING_KEYS` - `setup_s` (building and verifying the scene), `policy_s` (time
  inside policy calls), `sim_s` (time executing actions), `total_s` (the whole unit) - each a finite
  number of seconds, never negative. A phase the unit never reached took 0.0 s: every key is
  always there, so no path can leave one out.
- `rollout_mp4_sha256`: filled by `write` itself, the sha256 of `rollout.mp4` in the output
  directory when that file exists and `null` otherwise, so a fork writes its video before the
  result. `read` checks a `rollout.mp4` it finds beside the result against it; a result copied
  without its video still reads.
- `error`: what went wrong, as text, or `null`.
- `stub_policy`: `null`, or `zero` or `replay` when the unit was run against a stub
  (`zerowam_protocol.stubs`) instead of a submission. The competition refuses to score such a
  result outside a dry run (Q4).
- `served`: `null`, or the mapping the server's reply to `hello` said it served (protocol 3), which
  the fork records unchanged: `zerowam_protocol.policy.SERVED_KEYS` (`family_sha256`,
  `family_version`, `knobs`, `weights_fingerprint`, `weights_sha256`) and nothing else, `knobs` a
  mapping as `policy.checked_served` holds it at `hello`, so what an operator's `--knobs` resolved
  to is in the record of every unit it produced.

`write` and `read` hold a result to the same rules, and every refusal is a `ValueError` that names
the field and the rule - a run directory that cannot be written or read included, so a fork maps one
class to its harness exit and nothing here reaches it as an `OSError`.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .conventions import _is_finite
from .policy import SERVED_KEYS

__all__ = [
    "OUTCOMES",
    "REQUIRED_KEYS",
    "RESULT_JSON",
    "RESULT_VERSION",
    "ROLLOUT_MP4",
    "STUB_POLICIES",
    "TIMING_KEYS",
    "VOID_CAUSES",
    "read",
    "write",
]

#: Bumped whenever a field changes. 2: every field below required, `timing` named, provenance and
#: the rollout's hash recorded, `demo_sha256` the whole-bundle digest, `stub_policy` and `served`.
RESULT_VERSION = 2

RESULT_JSON = "result.json"
#: The rollout video a fork writes beside `result.json`; `write` hashes it when it is there.
ROLLOUT_MP4 = "rollout.mp4"

OUTCOMES = ("success", "failure", "void")
VOID_CAUSES = ("harness", "runtime")
#: What `timing` holds, exactly: seconds spent building the scene, inside policy calls, executing
#: actions, and in the whole unit.
TIMING_KEYS = ("setup_s", "policy_s", "sim_s", "total_s")
#: The stubs a result may say it was produced with (`zerowam_protocol.stubs`: `ZeroPolicy`,
#: `ReplayPolicy`).
STUB_POLICIES = ("zero", "replay")

#: Every field of a result: `write` sets each one and `read` requires each one; `extra` may use
#: none of them.
REQUIRED_KEYS = (
    "result_version",
    "unit_id",
    "demo_sha256",
    "outcome",
    "void_cause",
    "task_config",
    "task_config_sha256",
    "fork_commit",
    "steps",
    "step_limit",
    "fingerprint_ok",
    "policy_calls",
    "timing",
    "rollout_mp4_sha256",
    "error",
    "stub_policy",
    "served",
)

_CHUNK = 1 << 20
_SHA256_HEX = frozenset("0123456789abcdef")


def write(
    out_dir: str | Path,
    *,
    unit_id: str,
    demo_sha256: str,
    outcome: str,
    task_config: str,
    task_config_sha256: str,
    fork_commit: str,
    timing: Mapping[str, float],
    void_cause: str | None = None,
    steps: int | None = None,
    step_limit: int | None = None,
    fingerprint_ok: bool | None = None,
    policy_calls: int | None = None,
    error: str | None = None,
    stub_policy: str | None = None,
    served: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `result.json` into `out_dir` and return it.

    `demo_sha256` is `bundle.digest(bundle_dir)` of the unit's bundle. `timing` holds exactly
    `TIMING_KEYS`. `rollout_mp4_sha256` is not an argument: it is the sha256 of `rollout.mp4` in
    `out_dir` if that file exists now, so write the video first. `extra` adds keys of the fork's
    own and is refused if it names a field of the schema. Everything is checked before anything is
    written, and a refusal is a `ValueError` naming the field and the rule, an `out_dir` that
    cannot be written included. Neither file is written or read through a symlink, as `read` holds
    them: a `result.json` that is one is refused, not written through to wherever it points.
    """
    record: dict[str, Any] = {
        "result_version": RESULT_VERSION,
        "unit_id": unit_id,
        "demo_sha256": demo_sha256,
        "outcome": outcome,
        "void_cause": void_cause,
        "task_config": task_config,
        "task_config_sha256": task_config_sha256,
        "fork_commit": fork_commit,
        "steps": _plain_int(steps),
        "step_limit": _plain_int(step_limit),
        "fingerprint_ok": _plain_bool(fingerprint_ok),
        "policy_calls": _plain_int(policy_calls),
        "timing": _plain_timing(timing),
        "rollout_mp4_sha256": None,
        "error": error,
        "stub_policy": stub_policy,
        "served": dict(served) if isinstance(served, Mapping) else served,
    }
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise ValueError(f"extra is a {type(extra).__name__}, not a mapping of other fields")
        taken = sorted(str(key) for key in extra if key in REQUIRED_KEYS)
        if taken:
            raise ValueError(
                f"extra passes {', '.join(taken)}: fields of the result, which write sets from "
                "its own arguments; extra adds other keys only (result v2)"
            )
        odd = sorted(repr(key) for key in extra if not isinstance(key, str))
        if odd:
            raise ValueError(f"extra has keys {', '.join(odd)}, not strings")
        record.update(extra)
    problems = _problems(record)
    if problems:
        raise ValueError(f"result: {'; '.join(problems)}")
    _json_text(record)  # refused now, before anything is written
    out = Path(out_dir)
    # Never through a symlink, as `read` refuses one and the video is refused here too: a link
    # would put this unit's result outside its run directory, or over another unit's (result v2).
    if (out / RESULT_JSON).is_symlink():
        raise ValueError(
            f"{out / RESULT_JSON}: a symlink; a run's result is its own file (result v2)"
        )
    record["rollout_mp4_sha256"] = _rollout_sha256(out)
    # A run directory that cannot be written is a ValueError too: a fork maps this module's one
    # error class to its harness exit, and an OSError from here would escape it (result v2).
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / RESULT_JSON).write_text(_json_text(record))
    except OSError as exc:
        raise ValueError(f"{out / RESULT_JSON}: cannot be written: {exc}") from None
    return record


def read(out_dir: str | Path) -> dict[str, Any]:
    """The `result.json` of a run directory, held to every rule `write` applies.

    A result of another `RESULT_VERSION`, a missing field, a field that breaks its rule (a void
    without a cause, a success with one, a timing key missing or extra, ...) is refused, and so is
    a `rollout.mp4` beside it whose sha256 is not the one recorded. Neither file is read through a
    symlink: a run directory holds its own result and its own video. Every refusal is a
    `ValueError` naming the file, the field and the rule; a file that cannot be read is one too.
    """
    path = Path(out_dir) / RESULT_JSON
    # Never through a symlink, as the video is not read through one either: a run directory holds
    # its own result, and a link would let one file answer for two units (result v2).
    if path.is_symlink():
        raise ValueError(f"{path}: a symlink; a run's result is its own file (result v2)")
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise ValueError(f"{path}: no such file") from None
    except OSError as exc:
        raise ValueError(f"{path}: cannot be read: {exc}") from None
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not JSON: {exc}") from None
    try:
        record = json.loads(text, parse_constant=_refuse_constant)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{path}: not JSON: {exc}") from None
    if not isinstance(record, dict):
        raise ValueError(f"{path}: result is not a JSON object")
    version = record.get("result_version")
    if type(version) is not int or version != RESULT_VERSION:
        raise ValueError(
            f"{path}: result version {version!r}; this end reads {RESULT_VERSION}: write it with "
            "a fork on this protocol"
        )
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"{path}: result is missing {', '.join(missing)} (result v2)")
    problems = _problems(record)
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    video = Path(out_dir) / ROLLOUT_MP4
    if video.exists() or video.is_symlink():
        actual = _rollout_sha256(Path(out_dir))
        if record["rollout_mp4_sha256"] is None:
            raise ValueError(
                f"{video}: the result records no rollout video, so this one was added after it was "
                "written (result v2)"
            )
        if actual != record["rollout_mp4_sha256"]:
            raise ValueError(
                f"{video}: sha256 {actual}, the result records {record['rollout_mp4_sha256']}"
            )
    return record


# -- the rules ----------------------------------------------------------------------------------


def _problems(record: Mapping[str, Any]) -> list[str]:
    """Every way a result's fields break the schema. `write` and `read` both apply it."""
    problems = []
    unit_id = record.get("unit_id")
    if not (isinstance(unit_id, str) and unit_id.strip()):
        problems.append(f"unit_id is {unit_id!r}, not the unit's id (result v2)")
    for field in ("demo_sha256", "task_config_sha256"):
        if not _is_sha256(record.get(field)):
            problems.append(f"{field} is {record.get(field)!r}, not a sha256 (result v2)")
    if record.get("rollout_mp4_sha256") is not None and not _is_sha256(
        record["rollout_mp4_sha256"]
    ):
        problems.append(
            f"rollout_mp4_sha256 is {record['rollout_mp4_sha256']!r}, not null or a sha256 "
            "(result v2)"
        )
    outcome, cause = record.get("outcome"), record.get("void_cause")
    if outcome not in OUTCOMES:
        problems.append(f"outcome is {outcome!r}, not one of {', '.join(OUTCOMES)} (Q6)")
    elif outcome == "void" and cause not in VOID_CAUSES:
        problems.append(f"a void needs void_cause in {', '.join(VOID_CAUSES)}, not {cause!r} (Q6)")
    elif outcome != "void" and cause is not None:
        problems.append(f"a {outcome} must not carry void_cause {cause!r}; only a void does (Q6)")
    for field in ("task_config", "fork_commit"):
        value = record.get(field)
        if not (isinstance(value, str) and value.strip()):
            problems.append(f"{field} is {value!r}, not a non-empty string (result v2)")
    for field, least in (("steps", 0), ("step_limit", 1), ("policy_calls", 0)):
        value = record.get(field)
        if value is not None and not (_is_int(value) and value >= least):
            problems.append(f"{field} is {value!r}, not null or an integer >= {least} (result v2)")
    fingerprint_ok = record.get("fingerprint_ok")
    if fingerprint_ok is not None and not isinstance(fingerprint_ok, bool):
        problems.append(f"fingerprint_ok is {fingerprint_ok!r}, not null or a boolean (result v2)")
    problems += _timing_problems(record.get("timing"))
    error = record.get("error")
    if error is not None and not isinstance(error, str):
        problems.append(f"error is {error!r}, not null or text (result v2)")
    stub = record.get("stub_policy")
    if stub is not None and stub not in STUB_POLICIES:
        problems.append(
            f"stub_policy is {stub!r}, not null or one of {', '.join(STUB_POLICIES)} (Q4)"
        )
    served = record.get("served")
    if served is not None:
        if not isinstance(served, dict):
            problems.append(
                f"served is {served!r}, not null or the mapping the server's hello reply carried "
                "(result v2)"
            )
        else:
            unknown = sorted(str(key) for key in served if key not in SERVED_KEYS)
            if unknown:
                problems.append(
                    f"served carries {', '.join(unknown)}; it holds {', '.join(SERVED_KEYS)} and "
                    "nothing else, as the reply to hello carried it (protocol 3)"
                )
            # The one value whose shape `policy.checked_served` fixes at hello, so a result that
            # breaks it was not recorded unchanged from a reply (protocol 3).
            if "knobs" in served and not isinstance(served["knobs"], dict):
                problems.append(
                    f"served['knobs'] is {served['knobs']!r}, not the mapping of resolved knobs "
                    "the reply to hello carried (protocol 3)"
                )
    return problems


def _timing_problems(timing: Any) -> list[str]:
    names = ", ".join(TIMING_KEYS)
    if not isinstance(timing, dict):
        return [f"timing is {timing!r}, not a mapping of {names} (result v2)"]
    problems = []
    wrong = [f"lacks {key}" for key in TIMING_KEYS if key not in timing]
    wrong += [f"has {key!r}" for key in sorted(map(str, timing)) if key not in TIMING_KEYS]
    if wrong:
        problems.append(
            f"timing {', '.join(wrong)}; it holds exactly {names}, 0.0 for a phase the unit never "
            "reached (result v2)"
        )
    for key in TIMING_KEYS:
        value = timing.get(key)
        # _is_finite, never math.isfinite: JSON carries an integer no double can hold, and
        # math.isfinite raises OverflowError on one - never a ValueError a fork can map (result v2).
        if key in timing and not (_is_finite(value) and value >= 0):
            problems.append(f"timing.{key} is {value!r}, not a number of seconds >= 0 (result v2)")
    return problems


# -- helpers ------------------------------------------------------------------------------------


def _rollout_sha256(out: Path) -> str | None:
    """The sha256 of `rollout.mp4` in `out`, or None when there is none."""
    video = out / ROLLOUT_MP4
    if video.is_symlink():
        raise ValueError(f"{video}: a symlink; a result's video is its own file (result v2)")
    if not video.exists():
        return None
    if not video.is_file():
        raise ValueError(f"{video}: not a file (result v2)")
    sha = hashlib.sha256()
    try:
        with open(video, "rb") as handle:
            while chunk := handle.read(_CHUNK):
                sha.update(chunk)
    except OSError as exc:
        raise ValueError(f"{video}: cannot be read: {exc}") from None
    return sha.hexdigest()


def _plain_int(value: Any) -> Any:
    """A numpy integer as a Python one, so it can be written; anything else as it is."""
    return int(value) if _is_int(value) else value


def _plain_bool(value: Any) -> Any:
    """A numpy boolean as a Python one; anything else as it is."""
    return bool(value) if isinstance(value, (bool, np.bool_)) else value


def _plain_timing(timing: Any) -> Any:
    """`timing` as a dict of Python floats, for the values that are numbers a float can hold.

    Anything else is passed through untouched, for `_timing_problems` to name: converting a value
    this runs before the checks, so it never raises on one (`float(10 ** 400)` is an OverflowError).
    """
    if not isinstance(timing, Mapping):
        return timing
    return {key: float(value) if _is_finite(value) else value for key, value in timing.items()}


def _json_text(record: Mapping[str, Any]) -> str:
    try:
        return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"result is not plain JSON: {exc}") from None


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON number")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))
