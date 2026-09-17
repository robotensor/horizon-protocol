from __future__ import annotations

import json

import numpy as np
import pytest
from protocol_testing import demonstration, manifest, write_bundle

from zerowam_protocol import BundleError, bundle


def test_write_then_read_round_trips(tmp_path):
    record, arrays, _ = write_bundle(tmp_path / "unit")

    read_manifest, read_arrays = bundle.read(tmp_path / "unit")

    assert read_manifest == record
    assert read_manifest["bundle_version"] == bundle.BUNDLE_VERSION
    assert set(read_arrays) == set(arrays)
    for name, value in arrays.items():
        assert np.array_equal(read_arrays[name], value)


def test_manifest_hashes_the_arrays(tmp_path):
    record, _, _ = write_bundle(tmp_path / "unit")

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


def test_public_arrays_keeps_the_privileged_ones_out(tmp_path):
    _, arrays, _ = write_bundle(tmp_path / "unit")
    arrays["scene_fingerprint"] = np.zeros(4)
    arrays["success_condition"] = np.ones(2)

    public = bundle.public_arrays(arrays)

    assert "scene_fingerprint" not in public
    assert "success_condition" not in public
    assert {"frames_head", "frames_left_wrist", "times", "qpos", "endpose", "actions"} == set(
        public
    )


def test_private_stays_out_of_the_manifest(tmp_path):
    record, _, _ = write_bundle(tmp_path / "unit")

    assert (tmp_path / "unit" / bundle.PRIVATE_DIR / "scene.json").exists()
    assert (tmp_path / "unit" / bundle.PRIVATE_DIR / "expert.npz").exists()
    assert all(not name.startswith(bundle.PRIVATE_DIR) for name in record["files"])
