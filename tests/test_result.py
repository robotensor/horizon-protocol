from __future__ import annotations

import json

import numpy as np
import pytest
from protocol_testing import (
    TOO_BIG_FOR_A_DOUBLE,
    TOO_LONG_A_NAME,
    refuse_lookup,
    result_fields,
    write_bundle,
)

from vicl_protocol import bundle, result


def _rewrite(tmp_path, **changes):
    """Edit a written result.json in place, as a hand or another tool would."""
    path = tmp_path / result.RESULT_JSON
    record = json.loads(path.read_text())
    for key, value in changes.items():
        if value is _DROP:
            del record[key]
        else:
            record[key] = value
    path.write_text(json.dumps(record))


_DROP = object()


def test_write_then_read_round_trips(tmp_path):
    record = result.write(tmp_path, **result_fields())

    assert result.read(tmp_path) == record
    assert record["result_version"] == 2
    assert record["outcome"] == "success"
    assert record["void_cause"] is None
    assert record["rollout_mp4_sha256"] is None  # no video beside it
    assert record["stub_policy"] is None
    assert record["served"] is None
    assert set(record) == set(result.REQUIRED_KEYS)


def test_every_field_is_written_even_when_it_does_not_apply(tmp_path):
    """A void before the policy ran still says every field, null where it was never known."""
    record = result.write(
        tmp_path,
        **result_fields(
            outcome="void",
            void_cause="runtime",
            steps=None,
            step_limit=None,
            fingerprint_ok=None,
            policy_calls=None,
            timing={"setup_s": 3.0, "policy_s": 0.2, "sim_s": 0.0, "total_s": 3.3},
            error="PolicyUnavailable: prompt: the policy went away",
        ),
    )

    on_disk = json.loads((tmp_path / result.RESULT_JSON).read_text())
    assert set(on_disk) == set(result.REQUIRED_KEYS)
    assert on_disk["steps"] is None
    assert result.read(tmp_path) == record


def test_numpy_scalars_are_written_as_plain_numbers(tmp_path):
    record = result.write(
        tmp_path,
        **result_fields(
            steps=np.int64(12),
            policy_calls=np.int32(3),
            fingerprint_ok=np.bool_(True),
            timing={key: np.float32(1.5) for key in result.TIMING_KEYS},
        ),
    )

    assert record["steps"] == 12 and type(record["steps"]) is int
    assert record["fingerprint_ok"] is True
    assert record["timing"]["sim_s"] == 1.5
    assert result.read(tmp_path) == record


# -- extra never overwrites a field of the schema ----------------------------------------------


@pytest.mark.parametrize("field", result.REQUIRED_KEYS)
def test_extra_cannot_pass_a_field_of_the_schema(tmp_path, field):
    with pytest.raises(ValueError, match=f"extra passes {field}"):
        result.write(tmp_path, **result_fields(outcome="failure"), extra={field: "anything"})
    assert not (tmp_path / result.RESULT_JSON).exists()


def test_extra_cannot_turn_a_failure_into_a_success(tmp_path):
    with pytest.raises(ValueError, match="extra passes outcome"):
        result.write(tmp_path, **result_fields(outcome="failure"), extra={"outcome": "success"})


def test_extra_cannot_turn_a_failure_into_a_void(tmp_path):
    with pytest.raises(ValueError, match="extra passes outcome, void_cause"):
        result.write(
            tmp_path,
            **result_fields(outcome="failure"),
            extra={"outcome": "success", "void_cause": "model"},
        )


def test_extra_adds_keys_of_the_forks_own(tmp_path):
    record = result.write(tmp_path, **result_fields(), extra={"task_category": "press_push"})

    assert record["task_category"] == "press_push"
    assert result.read(tmp_path)["task_category"] == "press_push"


@pytest.mark.parametrize("extra", [{"when": float("nan")}, {"what": object()}, {1: "x"}, ["x"]])
def test_extra_that_is_not_plain_json_is_refused(tmp_path, extra):
    with pytest.raises(ValueError):
        result.write(tmp_path, **result_fields(), extra=extra)
    assert not (tmp_path / result.RESULT_JSON).exists()


# -- outcome and void_cause (Q6), at write and read ----------------------------------------------


def test_a_void_needs_a_cause(tmp_path):
    with pytest.raises(ValueError, match="void_cause.*Q6"):
        result.write(tmp_path, **result_fields(outcome="void"))

    with pytest.raises(ValueError, match="void_cause"):
        result.write(tmp_path, **result_fields(outcome="void", void_cause="cosmic-ray"))

    record = result.write(tmp_path, **result_fields(outcome="void", void_cause="harness"))
    assert record["void_cause"] == "harness"


def test_only_a_void_carries_a_cause(tmp_path):
    with pytest.raises(ValueError, match="must not carry"):
        result.write(tmp_path, **result_fields(outcome="failure", void_cause="harness"))


def test_unknown_outcome_is_refused(tmp_path):
    with pytest.raises(ValueError, match="outcome"):
        result.write(tmp_path, **result_fields(outcome="crashed"))


@pytest.mark.parametrize(
    "outcome, void_cause",
    [
        ("void", None),  # a void with no cause
        ("void", "model"),  # a cause nobody defined
        ("success", "harness"),  # a success that says it was void
        ("failure", "runtime"),
        ("crashed", None),
    ],
)
def test_read_applies_the_outcome_rule_write_applies(tmp_path, outcome, void_cause):
    result.write(tmp_path, **result_fields(outcome="failure"))
    _rewrite(tmp_path, outcome=outcome, void_cause=void_cause)

    with pytest.raises(ValueError, match="Q6"):
        result.read(tmp_path)


# -- the other fields -----------------------------------------------------------------------------


@pytest.mark.parametrize("unit_id", ["", "   ", None, 7])
def test_unit_id_is_required_and_not_empty(tmp_path, unit_id):
    with pytest.raises(ValueError, match="unit_id"):
        result.write(tmp_path, **result_fields(unit_id=unit_id))

    result.write(tmp_path, **result_fields())
    _rewrite(tmp_path, unit_id=unit_id)
    with pytest.raises(ValueError, match="unit_id"):
        result.read(tmp_path)


@pytest.mark.parametrize("field", ["demo_sha256", "task_config_sha256"])
@pytest.mark.parametrize("value", ["a" * 63, "A" * 64, "g" * 64, None, 12])
def test_the_hashes_are_sha256s(tmp_path, field, value):
    with pytest.raises(ValueError, match=f"{field} is .* not a sha256"):
        result.write(tmp_path, **result_fields(**{field: value}))

    result.write(tmp_path, **result_fields())
    _rewrite(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        result.read(tmp_path)


@pytest.mark.parametrize("field", ["task_config", "fork_commit"])
@pytest.mark.parametrize("value", ["", " ", None, 3])
def test_the_provenance_is_recorded(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        result.write(tmp_path, **result_fields(**{field: value}))


def test_the_provenance_is_a_required_argument(tmp_path):
    fields = result_fields()
    del fields["fork_commit"]
    with pytest.raises(TypeError, match="fork_commit"):
        result.write(tmp_path, **fields)


@pytest.mark.parametrize(
    "field, value",
    [
        ("steps", -1),
        ("steps", 2.5),
        ("steps", True),
        ("step_limit", 0),
        ("policy_calls", "7"),
        ("fingerprint_ok", "yes"),
        ("fingerprint_ok", 1),
        ("error", 42),
    ],
)
def test_counts_flags_and_the_error_have_their_types(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        result.write(tmp_path, **result_fields(**{field: value}))

    result.write(tmp_path, **result_fields())
    _rewrite(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        result.read(tmp_path)


# -- timing ---------------------------------------------------------------------------------------


def test_timing_holds_exactly_the_four_named_keys(tmp_path):
    assert result.TIMING_KEYS == ("setup_s", "policy_s", "sim_s", "total_s")
    timing = dict(result_fields()["timing"])

    for missing in result.TIMING_KEYS:
        partial = {k: v for k, v in timing.items() if k != missing}
        with pytest.raises(ValueError, match=f"timing lacks {missing}"):
            result.write(tmp_path, **result_fields(timing=partial))

    with pytest.raises(ValueError, match="timing has 'render_s'"):
        result.write(tmp_path, **result_fields(timing={**timing, "render_s": 1.0}))


@pytest.mark.parametrize(
    "value",
    [-0.1, float("nan"), float("inf"), "1.0", None, True, TOO_BIG_FOR_A_DOUBLE],
)
def test_timing_values_are_seconds(tmp_path, value):
    """The last one is a number JSON carries that no double holds: `float(it)` is an OverflowError,
    which is no ValueError, so `write` must range-test it before it converts (result v2)."""
    timing = {**result_fields()["timing"], "sim_s": value}
    with pytest.raises(ValueError, match="timing.sim_s") as refused:
        result.write(tmp_path, **result_fields(timing=timing))
    assert not isinstance(refused.value, ArithmeticError)


@pytest.mark.parametrize(
    "timing",
    [
        {"setup_s": 1.0, "policy_s": 2.0, "total_s": 3.0},  # the forks never wrote sim_s
        {"setup_s": 1.0, "policy_s": 2.0, "sim_s": 0.0, "total_s": -3.0},
        {"setup_s": 1.0, "policy_s": 2.0, "sim_s": 0.0, "total_s": 3.0, "extra_s": 1.0},
        [1.0, 2.0, 3.0, 4.0],
        {"setup_s": TOO_BIG_FOR_A_DOUBLE, "policy_s": 2.0, "sim_s": 0.0, "total_s": 3.0},
    ],
)
def test_read_checks_timing_as_write_does(tmp_path, timing):
    result.write(tmp_path, **result_fields())
    _rewrite(tmp_path, timing=timing)

    with pytest.raises(ValueError, match="timing") as refused:
        result.read(tmp_path)
    # An OverflowError is an ArithmeticError: a fork mapping this module's one class would miss it.
    assert not isinstance(refused.value, ArithmeticError)


def test_read_refuses_a_result_that_is_a_symlink(tmp_path):
    """A run directory holds its own result, as it holds its own video (result v2)."""
    run = tmp_path / "run"
    result.write(run, **result_fields())
    elsewhere = tmp_path / "somewhere_else.json"
    (run / result.RESULT_JSON).rename(elsewhere)
    (run / result.RESULT_JSON).symlink_to(elsewhere)

    with pytest.raises(ValueError, match="symlink"):
        result.read(run)


def test_write_refuses_a_result_that_is_a_symlink_and_writes_nothing_through_it(tmp_path):
    """`write` holds `result.json` to the rule `read` does, so no link carries it out of the run."""
    run = tmp_path / "run"
    run.mkdir()
    elsewhere = tmp_path / "elsewhere.json"
    (run / result.RESULT_JSON).symlink_to(elsewhere)

    with pytest.raises(ValueError, match="result.json: a symlink"):
        result.write(run, **result_fields())
    assert not elsewhere.exists()


@pytest.mark.parametrize("call", ["write", "read"])
def test_a_run_directory_the_os_will_not_look_up_is_a_value_error_not_an_oserror(tmp_path, call):
    """ENAMETOOLONG, as root too; EACCES (no search permission) takes the same lines."""
    run = tmp_path / TOO_LONG_A_NAME / "run"
    with pytest.raises(ValueError, match="cannot be (written|read)"):
        if call == "write":
            result.write(run, **result_fields())
        else:
            result.read(run)


@pytest.mark.parametrize("call", ["write", "read"])
@pytest.mark.parametrize("name", [result.RESULT_JSON, result.ROLLOUT_MP4])
def test_a_run_file_the_os_will_not_look_up_is_a_value_error(tmp_path, monkeypatch, name, call):
    """A file listed in a directory that refuses its lookup (mode r--, any user but root)."""
    (tmp_path / result.ROLLOUT_MP4).write_bytes(b"not really a video")
    result.write(tmp_path, **result_fields())
    refuse_lookup(monkeypatch, name)

    with pytest.raises(ValueError, match="cannot be (written|read)"):
        if call == "write":
            result.write(tmp_path, **result_fields())
        else:
            result.read(tmp_path)


def test_read_refuses_a_value_json_cannot_hold(tmp_path):
    result.write(tmp_path, **result_fields())
    path = tmp_path / result.RESULT_JSON
    path.write_text(path.read_text().replace('"sim_s": 120.0', '"sim_s": NaN'))

    with pytest.raises(ValueError, match="not JSON"):
        result.read(tmp_path)


# -- the rollout video ----------------------------------------------------------------------------


def test_write_hashes_the_rollout_video_beside_it(tmp_path):
    video = tmp_path / result.ROLLOUT_MP4
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42 a rollout")

    record = result.write(tmp_path, **result_fields())

    assert record["rollout_mp4_sha256"] == bundle.digest(video)
    assert result.read(tmp_path) == record


def test_a_run_directory_that_cannot_be_written_is_a_value_error(tmp_path):
    """A fork maps this module's one class to its harness exit: an OSError would escape it."""
    a_file = tmp_path / "afile"
    a_file.write_text("not a run directory")

    with pytest.raises(ValueError, match="cannot be written") as refused:
        result.write(a_file, **result_fields())
    assert not isinstance(refused.value, OSError)

    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / result.RESULT_JSON).mkdir()
    with pytest.raises(ValueError, match="cannot be written"):
        result.write(taken, **result_fields())


def test_the_rollout_hash_is_not_the_callers(tmp_path):
    with pytest.raises(TypeError, match="rollout_mp4_sha256"):
        result.write(tmp_path, **result_fields(), rollout_mp4_sha256="b" * 64)
    with pytest.raises(ValueError, match="extra passes rollout_mp4_sha256"):
        result.write(tmp_path, **result_fields(), extra={"rollout_mp4_sha256": "b" * 64})


def test_read_refuses_a_rollout_video_changed_after_the_result(tmp_path):
    video = tmp_path / result.ROLLOUT_MP4
    video.write_bytes(b"the rollout")
    result.write(tmp_path, **result_fields())
    video.write_bytes(b"another rollout")

    with pytest.raises(ValueError, match="rollout.mp4: sha256"):
        result.read(tmp_path)


def test_read_refuses_a_rollout_video_added_after_the_result(tmp_path):
    result.write(tmp_path, **result_fields())
    (tmp_path / result.ROLLOUT_MP4).write_bytes(b"a video from somewhere else")

    with pytest.raises(ValueError, match="records no rollout video"):
        result.read(tmp_path)


def test_a_result_copied_without_its_video_still_reads(tmp_path):
    video = tmp_path / result.ROLLOUT_MP4
    video.write_bytes(b"the rollout")
    record = result.write(tmp_path, **result_fields())
    video.unlink()

    assert result.read(tmp_path) == record


def test_a_symlinked_rollout_video_is_refused(tmp_path):
    (tmp_path / "elsewhere.mp4").write_bytes(b"the rollout")
    (tmp_path / result.ROLLOUT_MP4).symlink_to(tmp_path / "elsewhere.mp4")

    with pytest.raises(ValueError, match="symlink"):
        result.write(tmp_path, **result_fields())


# -- demo_sha256 is the whole bundle's digest -----------------------------------------------------


def test_demo_sha256_is_the_bundle_digest(tmp_path):
    record, *_ = write_bundle(tmp_path / "unit")
    digest = bundle.digest(tmp_path / "unit")

    written = result.write(
        tmp_path / "run", **result_fields(unit_id=record["unit_id"], demo_sha256=digest)
    )

    assert written["demo_sha256"] == digest == bundle.digest(tmp_path / "unit" / bundle.DEMO_JSON)
    assert written["demo_sha256"] != record["files"][bundle.FRAMES_NPZ]


# -- stub_policy (Q4) and served ------------------------------------------------------------------


@pytest.mark.parametrize("stub", result.STUB_POLICIES)
def test_a_stub_run_records_its_stub(tmp_path, stub):
    record = result.write(tmp_path, **result_fields(stub_policy=stub))

    assert result.read(tmp_path)["stub_policy"] == stub == record["stub_policy"]


def test_an_unknown_stub_is_refused(tmp_path):
    with pytest.raises(ValueError, match="stub_policy.*Q4"):
        result.write(tmp_path, **result_fields(stub_policy="random"))

    result.write(tmp_path, **result_fields())
    _rewrite(tmp_path, stub_policy="Zero")
    with pytest.raises(ValueError, match="stub_policy"):
        result.read(tmp_path)


def test_what_was_served_is_recorded_unchanged(tmp_path):
    served = {
        "family_sha256": "d" * 64,
        "family_version": "2026.09.1",
        "knobs": {"steps": 4},
        "weights_fingerprint": "e" * 64,
        "weights_sha256": "c" * 64,
    }

    record = result.write(tmp_path, **result_fields(served=served))

    assert record["served"] == served
    assert result.read(tmp_path)["served"] == served


@pytest.mark.parametrize("served", ["zerowam", ["zerowam"], {"knobs": {1, 2}}])
def test_served_is_a_mapping_of_plain_json(tmp_path, served):
    with pytest.raises(ValueError, match="served|JSON"):
        result.write(tmp_path, **result_fields(served=served))


def test_served_holds_the_keys_the_hello_reply_carries(tmp_path):
    """A result records what it was told, and only what protocol 3 defines (SERVED_KEYS)."""
    with pytest.raises(ValueError, match="served carries checkpoint, family"):
        result.write(tmp_path, **result_fields(served={"family": "zerowam", "checkpoint": "x"}))

    result.write(tmp_path, **result_fields(served={"knobs": {"steps": 4}}))  # a subset is fine
    _rewrite(tmp_path, served={"knobs": {"steps": 4}, "family": "zerowam"})
    with pytest.raises(ValueError, match="served carries family"):
        result.read(tmp_path)


@pytest.mark.parametrize("knobs", [[], 5, "steps=4", None])
def test_served_knobs_is_a_mapping_as_hello_holds_it(tmp_path, knobs):
    """`policy.checked_served` refuses these at hello, so no result may record one (protocol 3)."""
    with pytest.raises(ValueError, match=r"served\['knobs'\]"):
        result.write(tmp_path, **result_fields(served={"knobs": knobs}))

    result.write(tmp_path, **result_fields(served={"knobs": {"steps": 4}}))
    _rewrite(tmp_path, served={"knobs": knobs})
    with pytest.raises(ValueError, match=r"served\['knobs'\]"):
        result.read(tmp_path)


# -- reading --------------------------------------------------------------------------------------


@pytest.mark.parametrize("version", [1, 3, "2", True, None])
def test_read_refuses_any_other_version(tmp_path, version):
    result.write(tmp_path, **result_fields())
    _rewrite(tmp_path, result_version=version)

    with pytest.raises(ValueError, match="result version"):
        result.read(tmp_path)


@pytest.mark.parametrize("field", result.REQUIRED_KEYS[1:])
def test_read_refuses_a_file_missing_a_field(tmp_path, field):
    result.write(tmp_path, **result_fields())
    _rewrite(tmp_path, **{field: _DROP})

    with pytest.raises(ValueError, match=f"missing {field}"):
        result.read(tmp_path)


def test_read_refuses_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        result.read(tmp_path)


def test_read_refuses_what_is_not_a_json_object(tmp_path):
    path = tmp_path / result.RESULT_JSON
    for text, match in [(b"{", "not JSON"), (b"\xff\xfe", "not JSON"), (b"[1, 2]", "object")]:
        path.write_bytes(text)
        with pytest.raises(ValueError, match=match):
            result.read(tmp_path)


def test_read_refuses_a_result_that_is_not_a_file(tmp_path):
    (tmp_path / result.RESULT_JSON).mkdir()

    with pytest.raises(ValueError, match="cannot be read"):
        result.read(tmp_path)
