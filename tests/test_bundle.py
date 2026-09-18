from __future__ import annotations

import json

import numpy as np
import pytest
from protocol_testing import demonstration, expert, manifest, write_bundle

from zerowam_protocol import BundleError, BundleSchemaError, bundle


def test_write_then_read_round_trips(tmp_path):
    record, arrays, _, _ = write_bundle(tmp_path / "unit")

    read_manifest, read_arrays = bundle.read(tmp_path / "unit")

    assert read_manifest == record
    assert read_manifest["bundle_version"] == bundle.BUNDLE_VERSION
    assert set(read_arrays) == set(arrays)
    for name, value in arrays.items():
        assert np.array_equal(read_arrays[name], value)


def test_manifest_hashes_the_arrays(tmp_path):
    record, _, _, _ = write_bundle(tmp_path / "unit")

    assert record["files"][bundle.FRAMES_NPZ] == bundle.digest(
        tmp_path / "unit" / bundle.FRAMES_NPZ
    )


def test_read_refuses_a_tampered_npz(tmp_path):
    write_bundle(tmp_path / "unit")
    path = tmp_path / "unit" / bundle.FRAMES_NPZ
    path.write_bytes(path.read_bytes() + b"\0")

    with pytest.raises(BundleError, match="sha256"):
        bundle.read(tmp_path / "unit")

    manifest_read, _ = bundle.read(tmp_path / "unit", verify=False)
    assert manifest_read["task"] == "click_bell"


def test_read_refuses_another_bundle_version(tmp_path):
    write_bundle(tmp_path / "unit")
    path = tmp_path / "unit" / bundle.DEMO_JSON
    record = json.loads(path.read_text())
    record["bundle_version"] = bundle.BUNDLE_VERSION + 1
    path.write_text(json.dumps(record))

    with pytest.raises(BundleError, match="bundle version"):
        bundle.read(tmp_path / "unit")


def test_read_refuses_a_manifest_missing_a_required_key(tmp_path):
    write_bundle(tmp_path / "unit")
    path = tmp_path / "unit" / bundle.DEMO_JSON
    record = json.loads(path.read_text())
    del record["fingerprint_sha256"]
    path.write_text(json.dumps(record))

    with pytest.raises(BundleError, match="fingerprint_sha256"):
        bundle.read(tmp_path / "unit")


def test_write_refuses_a_demonstration_without_frames_or_times(tmp_path):
    arrays, _ = demonstration()

    with pytest.raises(BundleError, match="frames_"):
        bundle.write(
            tmp_path / "a",
            manifest=manifest(),
            arrays={"times": arrays["times"]},
        )
    with pytest.raises(BundleError, match="times"):
        bundle.write(
            tmp_path / "b",
            manifest=manifest(),
            arrays={"frames_head": arrays["frames_head"]},
        )


def test_write_refuses_a_manifest_missing_a_required_key(tmp_path):
    arrays, _ = demonstration()
    incomplete = manifest()
    del incomplete["task"]

    with pytest.raises(BundleError, match="task"):
        bundle.write(tmp_path / "unit", manifest=incomplete, arrays=arrays)


def test_the_allow_list_is_frames_and_times():
    """Decision Q4: a policy is given a demonstration's video and its times, and nothing else."""
    assert bundle.PUBLIC_PREFIXES == ("frames_",)
    assert bundle.PUBLIC_NAMES == ("times",)


def test_public_arrays_keeps_the_demonstrators_record_out():
    arrays, _ = demonstration()
    holding = {**arrays, **expert(), "scene_fingerprint": np.zeros(4), "states": np.zeros((6, 3))}

    public = bundle.public_arrays(holding)

    assert set(public) == {"frames_head", "frames_left_wrist", "times"}
    assert all(public[name] is arrays[name] for name in public)


@pytest.mark.parametrize(
    "name", ["qpos", "endpose", "actions", "ee_actions", "states", "demo_joints", "gripper_track"]
)
def test_write_refuses_any_array_outside_the_allow_list(tmp_path, name):
    arrays, _ = demonstration()
    arrays[name] = np.zeros((6, 16))

    with pytest.raises(BundleSchemaError, match="Q4") as refused:
        bundle.write(tmp_path / "unit", manifest=manifest(), arrays=arrays)
    assert repr(name) in str(refused.value)
    assert not (tmp_path / "unit" / bundle.FRAMES_NPZ).exists()


def test_read_refuses_a_bundle_whose_npz_holds_a_private_array(tmp_path):
    """Even with every hash in order: the rule is the allow-list, not the manifest."""
    write_bundle(tmp_path / "unit")
    unit = tmp_path / "unit"
    with np.load(unit / bundle.FRAMES_NPZ) as data:
        leaked = {name: data[name] for name in data.files}
    leaked["endpose"] = expert()["endpose"]
    np.savez_compressed(unit / bundle.FRAMES_NPZ, **leaked)
    record = json.loads((unit / bundle.DEMO_JSON).read_text())
    record["files"][bundle.FRAMES_NPZ] = bundle.digest(unit / bundle.FRAMES_NPZ)
    (unit / bundle.DEMO_JSON).write_text(json.dumps(record))

    with pytest.raises(BundleSchemaError, match="'endpose'.*Q4"):
        bundle.read(unit)


def test_a_schema_error_is_a_bundle_error_and_a_value_error():
    """A fork catches BundleSchemaError first (exit 2); any other BundleError is exit 4."""
    assert issubclass(BundleSchemaError, BundleError)
    assert issubclass(BundleSchemaError, ValueError)


def test_private_stays_out_of_the_manifest(tmp_path):
    record, _, _, _ = write_bundle(tmp_path / "unit")

    assert (tmp_path / "unit" / bundle.PRIVATE_DIR / "scene.json").exists()
    assert (tmp_path / "unit" / bundle.PRIVATE_DIR / "expert.npz").exists()
    assert all(not name.startswith(bundle.PRIVATE_DIR) for name in record["files"])
