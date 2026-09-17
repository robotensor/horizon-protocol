from __future__ import annotations

import json

import pytest

from zerowam_protocol import result


def test_write_then_read_round_trips(tmp_path):
    record = result.write(
        tmp_path,
        unit_id="rts-click_bell-000",
        demo_sha256="a" * 64,
        outcome="success",
        steps=214,
        step_limit=400,
        fingerprint_ok=True,
        policy_calls=7,
        timing={"setup_s": 1.5, "policy_s": 60.0, "sim_s": 120.0},
    )

    assert result.read(tmp_path) == record
    assert record["outcome"] == "success"
    assert record["void_cause"] is None


def test_a_void_needs_a_cause(tmp_path):
    with pytest.raises(ValueError, match="void_cause"):
        result.write(tmp_path, unit_id="u", demo_sha256="a" * 64, outcome="void")

    with pytest.raises(ValueError, match="void_cause"):
        result.write(
            tmp_path, unit_id="u", demo_sha256="a" * 64, outcome="void", void_cause="cosmic-ray"
        )

    record = result.write(
        tmp_path, unit_id="u", demo_sha256="a" * 64, outcome="void", void_cause="harness"
    )
    assert record["void_cause"] == "harness"


def test_only_a_void_carries_a_cause(tmp_path):
    with pytest.raises(ValueError, match="must not carry"):
        result.write(
            tmp_path,
            unit_id="u",
            demo_sha256="a" * 64,
            outcome="failure",
            void_cause="harness",
        )


def test_unknown_outcome_is_refused(tmp_path):
    with pytest.raises(ValueError, match="outcome"):
        result.write(tmp_path, unit_id="u", demo_sha256="a" * 64, outcome="crashed")


def test_read_refuses_a_file_missing_a_required_field(tmp_path):
    result.write(tmp_path, unit_id="u", demo_sha256="a" * 64, outcome="failure")
    path = tmp_path / result.RESULT_JSON
    record = json.loads(path.read_text())
    del record["demo_sha256"]
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="demo_sha256"):
        result.read(tmp_path)


def test_read_refuses_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="no such file"):
        result.read(tmp_path)
